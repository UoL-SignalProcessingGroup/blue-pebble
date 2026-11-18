"""Eigenray propagation with Bellhop (``bellhopcxx``) in arrivals mode.

Bellhop is an external program, not a Python dependency: install ``bellhopcxx`` from the
bellhopcuda project (https://github.com/A-New-BellHope/bellhopcuda) and put it on ``PATH``, or
give its full path as ``exe_path``.
"""

import os
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property

from ..environment import Bathymetry
from .acoustic import (
    AcousticPropagationModel,
    PropagationLoss,
    PropagationResult,
    SpectrumPropagationModel,
    SpectrumResult,
    _get_source_position,
    _get_source_tonal_arrays,
)

if TYPE_CHECKING:
    from stonesoup.types.state import State

    from ...platform import PlatformState

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]


def _warn(message: str, category: type[Warning]) -> None:
    """Warn, attributed to the first caller outside this package.

    The 2D and 3D models warn from inside nested ``__init__`` calls, so a fixed ``stacklevel``
    would point at the package for one of them, where a ``DeprecationWarning`` is hidden by
    default.
    """
    package_dir = str(Path(__file__).parent)
    frame, level = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(package_dir):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


@dataclass(frozen=True)
class Eigenray:
    """One arrival returned by Bellhop in arrivals mode.

    A single geometric path usually comes back as two or more arrivals from neighbouring
    beams, whose amplitudes add up to the path's amplitude, so sum every arrival rather than
    picking one per path.

    Attributes
    ----------
    amplitude : float
        Linear pressure amplitude (not dB), relative to the pressure 1 m from the source.
    phase_rad : float
        Phase of the arrival in radians, as Bellhop reports it: for example, pi for a path
        with one reflection off the pressure-release surface.
    delay_s : float
        One-way travel time in seconds.
    src_angle_deg : float
        Ray angle at the source in degrees, as Bellhop reports it (positive is downward).
    rcv_angle_deg : float
        Ray angle at the receiver in degrees.
    n_surface_bounces : int
        Number of surface reflections along the path.
    n_bottom_bounces : int
        Number of bottom reflections along the path.
    src_azimuth_deg : float or None
        Ray bearing at the source in degrees, measured anticlockwise from +x; only 3D runs
        report it.
    rcv_azimuth_deg : float or None
        Ray bearing at the receiver in degrees; only 3D runs report it.

    """

    amplitude: float
    phase_rad: float
    delay_s: float
    src_angle_deg: float
    rcv_angle_deg: float
    n_surface_bounces: int
    n_bottom_bounces: int
    src_azimuth_deg: float | None = None
    rcv_azimuth_deg: float | None = None


class BellhopAcousticPropagationModel(AcousticPropagationModel, SpectrumPropagationModel):
    r"""Ray propagation computed by Bellhop, as arrivals or per-sensor transfer functions.

    Each Bellhop run writes an environment (``.env``) and range-dependent bathymetry
    (``.bty``) file for the vertical plane through the source, runs ``bellhopcxx`` in arrivals
    mode with geometric beams, and parses the arrivals (``.arr``) file. The receivers are a
    rectilinear grid of every distinct receiver depth and range, from which each receiver's
    own point is picked. (``bellhopcxx`` sorts depths and ranges separately, so its irregular
    grid would pair them wrongly unless both happened to ascend together.)

    :meth:`propagate_spectrum` turns each sensor's arrivals into a transfer function, so the
    model can be the ``propagation_model`` of any simulator that accepts a
    :class:`~bluepebble.models.propagation.SpectrumPropagationModel`, in place of
    :class:`~bluepebble.models.propagation.rtrsAcousticPropagationModel`. For sensor
    :math:`n` with arrivals :math:`k` of amplitude :math:`a_k`, phase :math:`\phi_k` and delay
    :math:`\tau_k`,

    .. math::

        H_n(f) = \sum_k a_k \, e^{i \phi_k} \, e^{-i 2 \pi f \tau_k}, \qquad f > 0

    and :math:`H_n(-f) = H_n(f)^*` (a real channel). This is Blue Pebble's convention, in which
    a delay :math:`\tau` is :math:`e^{-i 2 \pi f \tau}`, and it is also the convention of
    Bellhop's reported phases: a bottom reflection's phase equals the phase of the Rayleigh
    reflection coefficient in it.

    Parameters
    ----------
    ssp : SoundSpeedProfile
        Sound speed profile, used as a function of depth only.
    bathymetry : Bathymetry
        Water depth, sampled along each run's vertical plane.
    far_field : bool, optional
        If True, make one run to the array reference point only and give every sensor its
        arrivals, each delayed by the sensor's plane-wave offset (see Assumptions). By default
        (False) Bellhop traces to every sensor.
    separate_planes : bool, optional
        When tracing to every sensor: if True, make one run per sensor, each in the vertical
        plane through that sensor. By default (False) one run covers the whole array, in the
        plane from the source towards the array reference point, with every sensor at its own
        horizontal range and depth. Far field has a single receiver and so a single plane, so
        ``separate_planes=True`` with ``far_field=True`` falls back to False, with a warning;
        the result is the same.
    plane_tolerance_deg : float, optional
        With ``separate_planes=True``: sensors whose vertical planes through the source are
        within this many degrees of bearing of each other share one run, in the plane at their
        mean bearing, each still at its own range and depth. The default, 0, shares a run only
        between sensors in exactly the same plane, such as sensors stacked at the same
        horizontal position, which leaves the result unchanged. A larger tolerance makes fewer
        runs; at the extreme it approaches the default, single plane.
    bathymetry_interpolation : str, optional
        How Bellhop joins the bathymetry points along each plane (or, in Bellhop3D, across its
        grid): ``"curvilinear"`` (the default) or ``"linear"``. Curvilinear takes the seabed's
        curvature into account when reflecting rays, so a curved seabed such as a seamount
        converges with a few hundred points; straight segments act as flat mirrors and need
        many more. On a flat or planar seabed the two give the same answer. Over noisy gridded
        data, curvature estimated from neighbouring points can amplify the noise, and linear
        may be the safer choice.
    max_workers : int or None, optional
        With ``separate_planes=True``: how many Bellhop runs go at once. The default, None,
        allows one per CPU core; 1 makes the runs one after another. Each run keeps
        ``bellhopcxx``'s own threads, so the arrivals are the same either way. (Forcing each
        run onto one thread would be slower, and changes how ``bellhopcxx`` merges nearly
        coincident arrivals, by tens of nanoseconds.)
    receiver_geometry : str, optional
        Deprecated; use ``far_field`` and ``separate_planes``. ``"reference"`` is the default,
        ``"per_sensor"`` is ``separate_planes=True`` and ``"far_field"`` is
        ``far_field=True``.
    use_all_frequencies : bool, optional
        For :meth:`propagate`: report the transmission loss at every tonal in the source's
        metadata, rather than only the loudest.
    exe_path : str, optional
        Name or full path of the ``bellhopcxx`` executable.
    n_ssp_samples : int, optional
        Number of depths in the sound speed table written to the environment file.
    n_bathy_samples : int, optional
        Number of ranges in the bathymetry profile.
    bottom_sound_speed_mps : float, optional
        Compressional sound speed of the sediment half-space, in m/s.
    bottom_density_g_cm3 : float, optional
        Density of the sediment half-space, in g/cm³.
    bottom_attenuation_db_per_wavelength : float, optional
        Compressional attenuation of the sediment half-space, in dB per wavelength.
    beam_angle_limits_deg : tuple of float, optional
        Lowest and highest launch angles of the beam fan, in degrees.
    num_beams : int, optional
        Number of beams in the fan; 0 lets Bellhop choose. In deep isovelocity water, 2000
        beams put the transfer function within 0.3% of the analytic Lloyd's mirror field
        where Bellhop's own choice was within 2%.
    memory : str, optional
        Memory ``bellhopcxx`` allocates per run (its ``--memory`` option).

    Assumptions
    -----------
    - Propagation is in a vertical plane through the source (2D), so
      out-of-plane bathymetry and refraction are ignored. By default the
      whole array shares the plane towards its reference point, which is exact when the
      bathymetry is the same on every bearing and close when the array's aperture is small
      against the range.
    - With ``far_field=True`` the source is taken to be in the far field of the array.
      Sensor :math:`n` gets the reference point's arrivals with each delay shifted by
      :math:`\Delta\tau_n = -(\mathbf{r}_n - \mathbf{r}_\text{ref}) \cdot \hat{\mathbf{u}} / c`,
      where :math:`\hat{\mathbf{u}}` is the straight-line unit vector from the reference
      point to the source and :math:`c` the sound speed at the reference depth. Every
      arrival is shifted alike, whatever its ray angle, and amplitudes and phases are the
      same at every sensor, so wavefront curvature and multipath differences across the
      array are ignored.
    - The sound speed varies with depth only: the same profile applies at every range.
    - The surface is a flat pressure-release (vacuum) boundary.
    - The bottom is a fluid half-space (no shear) below the bathymetry, with a flat-interface
      reflection coefficient and no roughness.
    - In :meth:`propagate_spectrum`, Bellhop runs once at the centre of the requested band, so
      each arrival's amplitude and phase (spreading and bottom loss) are taken as constant
      across the band; only the delay phase varies with frequency.

    """

    bathymetry: Bathymetry = Property(doc="Bathymetry model (water depth)")
    far_field: bool = Property(
        default=False,
        doc="Trace to the array reference point only, with plane-wave delays across the "
        "array, instead of to every sensor",
    )
    separate_planes: bool = Property(
        default=False,
        doc="When tracing to every sensor, one run per sensor in its own vertical plane, "
        "instead of one run in the plane towards the array reference point",
    )
    plane_tolerance_deg: float = Property(
        default=0.0,
        doc="With separate_planes, sensors whose planes are within this many degrees of "
        "bearing share one run; 0 shares only exactly coincident planes",
    )
    bathymetry_interpolation: str = Property(
        default="curvilinear",
        doc='How Bellhop joins the bathymetry points: "curvilinear" (default) or "linear"',
    )
    max_workers: int | None = Property(
        default=None,
        doc="With separate_planes, how many Bellhop runs go at once; None allows one per CPU "
        "core, 1 runs them one after another",
    )
    use_all_frequencies: bool = Property(
        default=False,
        doc="For propagate(): transmission loss at every source tonal, not only the loudest",
    )
    exe_path: str = Property(
        default="bellhopcxx", doc="Name or full path of the bellhopcxx executable"
    )
    n_ssp_samples: int = Property(
        default=50, doc="Number of depths in the sound speed table written to the env file"
    )
    n_bathy_samples: int = Property(
        default=50, doc="Number of ranges in the bathymetry profile written to the bty file"
    )
    bottom_sound_speed_mps: float = Property(
        default=1700.0, doc="Compressional sound speed of the sediment half-space, in m/s"
    )
    bottom_density_g_cm3: float = Property(
        default=1.5, doc="Density of the sediment half-space, in g/cm^3"
    )
    bottom_attenuation_db_per_wavelength: float = Property(
        default=0.5, doc="Compressional attenuation of the sediment half-space, in dB/wavelength"
    )
    beam_angle_limits_deg: tuple[float, float] = Property(
        default=(-89.0, 89.0), doc="Lowest and highest beam launch angles, in degrees"
    )
    num_beams: int = Property(
        default=2000,
        doc="Number of beams in the fan. Denser fans bracket each receiver more tightly, which "
        "sets the delay and transfer-function accuracy; 0 lets Bellhop choose.",
    )
    memory: str = Property(
        default="256MiB",
        doc="Memory bellhopcxx allocates per run (its --memory option, e.g. '256MiB'). Its "
        "own default, 4GiB, makes each run spend most of its time allocating.",
    )
    receiver_geometry: str | None = Property(
        default=None,
        doc='Deprecated; use far_field and separate_planes. "reference", "per_sensor" or '
        '"far_field".',
    )

    #: ``(far_field, separate_planes)`` for each deprecated ``receiver_geometry``.
    _legacy_geometries: dict[str, tuple[bool, bool]] = {
        "reference": (False, False),
        "per_sensor": (False, True),
        "far_field": (True, False),
    }

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the model and resolve ``far_field`` and ``separate_planes``.

        A deprecated ``receiver_geometry`` is translated to the two settings. A combination
        that means nothing falls back to the nearest one that does, with a warning.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        Raises
        ------
        ValueError
            If ``receiver_geometry`` has an unknown value, or is given together with
            ``far_field`` or ``separate_planes``; if ``plane_tolerance_deg`` is negative; if
            ``bathymetry_interpolation`` is unknown; or if ``max_workers`` is below 1.

        """
        super().__init__(*args, **kwargs)
        if not self.plane_tolerance_deg >= 0:
            raise ValueError(
                f"plane_tolerance_deg ({self.plane_tolerance_deg}) must be zero or positive."
            )
        if self.bathymetry_interpolation not in _BATHYMETRY_INTERPOLATIONS:
            raise ValueError(
                f"bathymetry_interpolation ({self.bathymetry_interpolation!r}) must be one of "
                f"{tuple(_BATHYMETRY_INTERPOLATIONS)}."
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(f"max_workers ({self.max_workers}) must be at least 1, or None.")
        if self.receiver_geometry is not None:
            if self.receiver_geometry not in self._legacy_geometries:
                raise ValueError(
                    f"receiver_geometry ({self.receiver_geometry!r}) must be one of "
                    f"{tuple(self._legacy_geometries)}."
                )
            if self.far_field or self.separate_planes:
                raise ValueError(
                    "Give either receiver_geometry (deprecated) or far_field and "
                    "separate_planes, not both."
                )
            self.far_field, self.separate_planes = self._legacy_geometries[self.receiver_geometry]
            replacement = (
                " and ".join(
                    f"{name}=True"
                    for name, value in (
                        ("far_field", self.far_field),
                        ("separate_planes", self.separate_planes),
                    )
                    if value
                )
                or "the defaults"
            )
            _warn(
                f"receiver_geometry is deprecated; use {replacement} instead of "
                f"receiver_geometry={self.receiver_geometry!r}.",
                DeprecationWarning,
            )
        self._resolve_planes()

    def _resolve_planes(self) -> None:
        """Fall back from ``separate_planes`` with ``far_field``, which has a single plane."""
        if self.separate_planes and self.far_field:
            _warn(
                "far_field=True traces to the reference point only, so there is one plane; "
                "ignoring separate_planes=True. The result is the same.",
                UserWarning,
            )
            self.separate_planes = False

    def compute_eigenrays(
        self,
        source_position: ArrayLike,
        receiver_position: ArrayLike,
        frequency_hz: float,
    ) -> list[Eigenray]:
        """Run Bellhop for one source and receiver and return every arrival.

        Parameters
        ----------
        source_position : ArrayLike
            World-frame ``(x, y, z)`` of the source in metres, with z negative below the
            surface (Blue Pebble's +z up convention).
        receiver_position : ArrayLike
            World-frame ``(x, y, z)`` of the receiver in metres, with the same convention.
        frequency_hz : float
            Frequency in Hz, which sets the bottom loss and beam widths.

        Returns
        -------
        list[Eigenray]
            Every arrival Bellhop found; empty if it found no paths.

        Raises
        ------
        ValueError
            If either point is not below the surface, or the two are at the same horizontal
            position.
        FileNotFoundError
            If ``bellhopcxx`` cannot be found.
        RuntimeError
            If ``bellhopcxx`` fails.

        """
        receiver = np.asarray(receiver_position, dtype=float).reshape(3, 1)
        return self._run_arrivals(source_position, receiver, receiver[:2, 0], frequency_hz)[0]

    def propagate_spectrum(
        self,
        platform: "PlatformState",
        source: "State",
        frequencies_hz: ArrayLike,
    ) -> SpectrumResult:
        """Return each sensor's transfer function and earliest-arrival delay.

        Parameters
        ----------
        platform : PlatformState
            Platform state providing the sensor array geometry.
        source : State
            Source state, whose metadata gives its ``position_mapping``.
        frequencies_hz : ArrayLike
            Frequencies at which to evaluate the transfer functions, in Hz.

        Returns
        -------
        SpectrumResult
            ``(H_sensors, sensor_delays_s)``: transfer functions of shape
            ``(num_sensors, num_frequencies)``, and each sensor's earliest arrival in seconds.
            A sensor Bellhop found no path to gets zeros and a NaN delay.

        """
        frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
        magnitudes = np.abs(frequencies)
        if not np.any(magnitudes > 0):
            raise ValueError("frequencies_hz must include a non-zero frequency.")
        # Simulators pass full complex FFT grids, so run at the centre of the |f| band.
        band_centre_hz = float((magnitudes.min() + magnitudes.max()) / 2)
        if self.far_field:
            # Every sensor hears the reference point's arrivals shifted by its plane-wave
            # offset, so its transfer function is the reference point's times a delay ramp:
            # one exponential per sensor and frequency, not one per arrival.
            rays, offsets_s = self._far_field_arrivals(platform, source, band_centre_hz)
            transfer_functions = _transfer_function(rays, frequencies)[np.newaxis, :] * np.exp(
                -2j * np.pi * offsets_s[:, np.newaxis] * frequencies[np.newaxis, :]
            )
            first_s = min((r.delay_s for r in rays), default=np.nan)
            return transfer_functions, first_s + offsets_s
        arrivals = self._sensor_arrivals(platform, source, band_centre_hz)
        transfer_functions = np.stack([_transfer_function(rays, frequencies) for rays in arrivals])
        delays_s = np.array(
            [min((r.delay_s for r in rays), default=np.nan) for rays in arrivals], dtype=float
        )
        return transfer_functions, delays_s

    def propagate(self, platform: "PlatformState", source: "State") -> PropagationResult:
        r"""Return the transmission loss and travel time to the array's reference point.

        The transmission loss is :math:`-20 \log_{10} |H(f)|` at the source's loudest tonal
        (or at every tonal, with ``use_all_frequencies``), from a Bellhop run at that
        frequency. The travel time is the earliest arrival, where
        :class:`~bluepebble.models.propagation.rtrsAcousticPropagationModel` uses the
        straight-line distance over the sound speed.

        Parameters
        ----------
        platform : PlatformState
            Platform state providing the array reference point.
        source : State
            Source state, whose metadata gives its position mapping and tonals.

        Returns
        -------
        PropagationResult
            ``(transmission_loss_db, travel_time_s)``. The loss is a float, or one value per
            tonal with ``use_all_frequencies``; 999 dB where Bellhop found no path.

        """
        tonal_frequencies, tonal_amplitudes = _get_source_tonal_arrays(source)
        if self.use_all_frequencies:
            frequencies = tonal_frequencies
        else:
            frequencies = tonal_frequencies[[int(np.argmax(tonal_amplitudes))]]
        source_position = _get_source_position(source)
        reference = np.asarray(platform.array.ref_state_vector, dtype=float).reshape(3, 1)

        losses_db = []
        travel_time_s = np.nan
        for frequency_hz in frequencies:
            rays = self._run_arrivals(
                source_position, reference, reference[:2, 0], float(frequency_hz)
            )[0]
            magnitude = abs(_transfer_function(rays, np.array([frequency_hz]))[0])
            losses_db.append(999.0 if magnitude < 1e-12 else float(-20 * np.log10(magnitude)))
            if np.isnan(travel_time_s):
                travel_time_s = min((r.delay_s for r in rays), default=np.nan)
        loss: PropagationLoss = (
            np.asarray(losses_db, dtype=np.float64) if self.use_all_frequencies else losses_db[0]
        )
        return loss, float(travel_time_s)

    def _sensor_arrivals(
        self, platform: "PlatformState", source: "State", frequency_hz: float
    ) -> list[list[Eigenray]]:
        """Return the arrivals at every sensor, per ``far_field`` and ``separate_planes``."""
        source_position = _get_source_position(source)
        sensors = np.asarray(platform.array.state_vector, dtype=float)
        if self.far_field:
            rays, offsets_s = self._far_field_arrivals(platform, source, frequency_hz)
            return [[replace(r, delay_s=r.delay_s + dt) for r in rays] for dt in offsets_s]
        if self.separate_planes:
            groups = self._plane_groups(source_position, sensors)
            workers = min(self.max_workers or os.cpu_count() or 1, len(groups))

            def run(group: NDArray[np.intp], plane_xy: FloatArray) -> list[list[Eigenray]]:
                return self._run_arrivals(
                    source_position, sensors[:, group], plane_xy, frequency_hz
                )

            if workers > 1:
                # Each run is a bellhopcxx process, so threads run them truly in parallel.
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(lambda g: run(*g), groups))
            else:
                results = [run(*g) for g in groups]
            arrivals: list[list[Eigenray]] = [[] for _ in range(sensors.shape[1])]
            for (group, _), group_arrivals in zip(groups, results, strict=True):
                for n, rays in zip(group, group_arrivals, strict=True):
                    arrivals[n] = rays
            return arrivals
        reference_xy = np.asarray(platform.array.ref_state_vector, dtype=float).ravel()[:2]
        return self._run_arrivals(source_position, sensors, reference_xy, frequency_hz)

    def _plane_groups(
        self, source_position: ArrayLike, sensors: FloatArray
    ) -> list[tuple[NDArray[np.intp], FloatArray]]:
        """Group sensors whose vertical planes through the source share a run.

        Returns ``(sensor indices, plane_xy)`` for each group, where ``plane_xy`` is a
        horizontal point on the group's plane, at its mean bearing from the source. Sensors
        are sorted by bearing, and a group runs until a sensor is more than
        ``plane_tolerance_deg`` past the group's first. With a tolerance of 0, only bearings
        equal to within 1e-9 degrees, that is the same plane, are grouped.
        """
        source_xy = np.asarray(source_position, dtype=float).ravel()[:2]
        dx, dy = sensors[0] - source_xy[0], sensors[1] - source_xy[1]
        # Bearings relative to the sensors' mean direction, so a group never straddles the
        # +-180 degree wrap.
        mean_rad = np.arctan2(dy.sum(), dx.sum())
        relative_deg = np.degrees(np.angle(np.exp(1j * (np.arctan2(dy, dx) - mean_rad))))
        tolerance_deg = max(self.plane_tolerance_deg, 1e-9)
        order = np.argsort(relative_deg, kind="stable")
        groups: list[list[int]] = []
        start_deg = -np.inf
        for n in order:
            if relative_deg[n] - start_deg > tolerance_deg:
                groups.append([])
                start_deg = relative_deg[n]
            groups[-1].append(int(n))
        result = []
        for group in groups:
            bearing_rad = mean_rad + np.radians(np.mean(relative_deg[group]))
            plane_xy = source_xy + np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
            result.append((np.array(group, dtype=np.intp), plane_xy))
        return result

    def _far_field_arrivals(
        self, platform: "PlatformState", source: "State", frequency_hz: float
    ) -> tuple[list[Eigenray], FloatArray]:
        """Return the reference point's arrivals and every sensor's plane-wave offset (s)."""
        source_position = _get_source_position(source)
        sensors = np.asarray(platform.array.state_vector, dtype=float)
        reference = np.asarray(platform.array.ref_state_vector, dtype=float).reshape(3, 1)
        (rays,) = self._run_arrivals(source_position, reference, reference[:2, 0], frequency_hz)
        return rays, self._plane_wave_offsets_s(source_position, sensors, reference[:, 0])

    def _plane_wave_offsets_s(
        self, source_position: ArrayLike, sensors: FloatArray, reference: FloatArray
    ) -> FloatArray:
        """Return each sensor's plane-wave delay relative to the reference point, in seconds.

        A sensor displaced towards the source hears the wavefront earlier, so its offset is
        negative.
        """
        to_source = np.asarray(source_position, dtype=float).ravel()[:3] - reference
        distance_m = float(np.linalg.norm(to_source))
        if distance_m <= 0:
            raise ValueError("The source must not be at the array reference point.")
        speed_mps = float(self.ssp.calculate(reference[2]))
        offsets_m = (sensors - reference[:, np.newaxis]).T @ (to_source / distance_m)
        return np.asarray(-offsets_m / speed_mps, dtype=float)

    def _run_arrivals(
        self,
        source_position: ArrayLike,
        receiver_positions: FloatArray,
        plane_towards_xy: FloatArray,
        frequency_hz: float,
    ) -> list[list[Eigenray]]:
        """Run Bellhop once for one source and several receivers.

        Parameters
        ----------
        source_position : ArrayLike
            World-frame source position, z negative below the surface.
        receiver_positions : FloatArray
            World-frame receiver positions, shape ``(3, num_receivers)``.
        plane_towards_xy : FloatArray
            A horizontal point that sets the vertical plane's bearing from the source.
        frequency_hz : float
            Frequency of the run in Hz.

        Returns
        -------
        list[list[Eigenray]]
            The arrivals at each receiver, in order.

        """
        source = np.asarray(source_position, dtype=float).ravel()
        receivers = np.asarray(receiver_positions, dtype=float).reshape(3, -1)
        source_depth_m = -source[2]
        receiver_depths_m = -receivers[2]
        if source_depth_m <= 0 or np.any(receiver_depths_m <= 0):
            raise ValueError(
                "Source and receivers must be below the surface (z < 0); got source z = "
                f"{source[2]} and receiver z = {receivers[2].tolist()}."
            )
        ranges_m = np.hypot(receivers[0] - source[0], receivers[1] - source[1])
        if np.any(ranges_m <= 0):
            raise ValueError("Source and receivers must be at different horizontal positions.")

        bearing = np.asarray(plane_towards_xy, dtype=float).ravel()[:2] - source[:2]
        if np.hypot(*bearing) <= 0:
            bearing = receivers[:2, int(np.argmax(ranges_m))] - source[:2]
        end_xy = source[:2] + bearing / np.hypot(*bearing) * float(ranges_m.max())

        # Bellhop sorts depths and ranges separately, so write each grid axis sorted and unique
        # (at the precision written) and look every receiver up in it.
        grid_depths_m, depth_index = np.unique(np.round(receiver_depths_m, 4), return_inverse=True)
        grid_ranges_km, range_index = np.unique(np.round(ranges_m / 1000, 6), return_inverse=True)

        exe = self._resolve_exe()
        bathy_range_km, bottom_depth_m = self._bathymetry_profile(
            source[:2], end_xy, float(ranges_m.max())
        )
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "bp"
            self._write_env(
                stem,
                source_depth_m,
                grid_depths_m,
                grid_ranges_km * 1000,
                float(bottom_depth_m.max()),
                frequency_hz,
            )
            self._write_bty(
                stem,
                bathy_range_km,
                bottom_depth_m,
                _BATHYMETRY_INTERPOLATIONS[self.bathymetry_interpolation],
            )
            self._run_bellhopcxx(exe, stem, self.memory)
            blocks = self._parse_arr(stem.with_suffix(".arr"))
        # Blocks run over depths, with range varying fastest.
        return [
            blocks[d * grid_ranges_km.size + r]
            for d, r in zip(depth_index.ravel(), range_index.ravel(), strict=True)
        ]

    def _resolve_exe(self) -> str:
        """Return the full path of ``bellhopcxx``, looking it up once."""
        cached = getattr(self, "_exe_resolved", None)
        if cached is None:
            cached = which(self.exe_path)
            if cached is None:
                raise FileNotFoundError(
                    f"bellhopcxx executable '{self.exe_path}' not found. Build bellhopcxx from "
                    "https://github.com/A-New-BellHope/bellhopcuda and put it on PATH, or "
                    "pass its full path as exe_path."
                )
            self._exe_resolved = cached
        return cached

    def _bathymetry_profile(
        self, source_xy_m: FloatArray, end_xy_m: FloatArray, range_m: float
    ) -> tuple[FloatArray, FloatArray]:
        """Sample the water depth along the line from the source to ``end_xy_m``.

        Returns
        -------
        tuple of FloatArray
            Ranges from the source in km, and positive water depths in metres.

        """
        t = np.linspace(0.0, 1.0, self.n_bathy_samples)
        points = source_xy_m[None, :] + t[:, None] * (end_xy_m - source_xy_m)[None, :]
        # Bathymetry models give the seabed as z below the surface (negative); Bellhop wants
        # positive depths.
        depth_m = np.array(
            [abs(float(self.bathymetry.get_depth(float(x), float(y)))) for x, y in points]
        )
        return t * range_m / 1000.0, depth_m

    def _write_env(
        self,
        stem: Path,
        source_depth_m: float,
        receiver_depths_m: FloatArray,
        receiver_ranges_m: FloatArray,
        water_depth_m: float,
        frequency_hz: float,
    ) -> None:
        """Write a Bellhop arrivals-mode environment file for a rectilinear receiver grid."""
        receiver_depths = np.atleast_1d(receiver_depths_m)
        ranges_km = np.atleast_1d(receiver_ranges_m) / 1000.0
        low_angle, high_angle = self.beam_angle_limits_deg
        env = "\n".join(
            [
                "'BellhopArrivals'",
                f"{frequency_hz:.4f}",
                *self._medium_lines(water_depth_m),
                "1",
                f"{source_depth_m:.4f} /",
                str(receiver_depths.size),
                " ".join(f"{d:.4f}" for d in receiver_depths) + " /",
                str(ranges_km.size),
                " ".join(f"{r:.6f}" for r in ranges_km) + " /",
                "'AG'",  # ASCII arrivals, geometric (hat) beams in Cartesian coordinates
                str(self.num_beams),
                f"{low_angle} {high_angle} /",
                f"0.0 {water_depth_m * 1.05:.2f} {ranges_km.max() * 1.05:.6f}",
            ]
        )
        stem.with_suffix(".env").write_text(env)

    def _medium_lines(self, water_depth_m: float) -> list[str]:
        """Return the environment file's medium, sound speed and bottom lines.

        Depths are written to the micrometre, like the bathymetry, so the deepest bathymetry
        point and the sound speed table's last depth are written from the same number and
        Bellhop never finds the seabed below the table.
        """
        depths = np.linspace(0.0, water_depth_m, self.n_ssp_samples)
        speeds = [float(self.ssp.calculate(d)) for d in depths]
        return [
            "1",  # number of media
            "'SVW'",  # spline SSP, vacuum above the surface, attenuation in dB/wavelength
            f"0 0.0 {water_depth_m:{_DEPTH_FORMAT}}",
            *(f"  {d:{_DEPTH_FORMAT}}  {c:.4f}  /" for d, c in zip(depths, speeds, strict=True)),
            "'A~' 0.0",  # acousto-elastic half-space below a .bty bathymetry, no roughness
            f"{water_depth_m:{_DEPTH_FORMAT}}  {self.bottom_sound_speed_mps}  0.0  "
            f"{self.bottom_density_g_cm3}  {self.bottom_attenuation_db_per_wavelength}  /",
        ]

    @staticmethod
    def _write_bty(
        stem: Path, range_km: FloatArray, depth_m: FloatArray, interpolation: str = "C"
    ) -> None:
        """Write a range-dependent bathymetry file.

        ``interpolation`` is Bellhop's code for joining the points: ``"C"`` curvilinear or
        ``"L"`` linear.

        Depths are written to the micrometre and ranges to the millimetre. Rounding depths to
        the centimetre put a kink at every point of a sloped seabed: with points metres apart
        that tilts segments by tenths of a degree, and the bottom-reflected arrivals then
        changed with the number of points rather than converging.
        """
        lines = [f"'{interpolation}'", str(len(range_km))]
        lines += [f"{r:.6f} {d:{_DEPTH_FORMAT}}" for r, d in zip(range_km, depth_m, strict=True)]
        stem.with_suffix(".bty").write_text("\n".join(lines))

    @staticmethod
    def _run_bellhopcxx(exe: str, stem: Path, memory: str, *flags: str) -> None:
        """Run ``bellhopcxx`` on ``stem``, raising with its output if it fails."""
        try:
            subprocess.run(
                [exe, *flags, f"--memory={memory}", str(stem)],
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"bellhopcxx failed (exit {e.returncode}).\n"
                f"env file: {stem}.env\nstdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e

    @staticmethod
    def _parse_arr(arr_path: Path) -> list[list[Eigenray]]:
        """Parse a 2D ASCII arrivals file for a rectilinear receiver grid.

        The file holds a header (``'2D'``, the frequency, the source depths, receiver depths
        and ranges, each count first, and a maximum-arrivals hint), then for every source,
        receiver depth and range (range varying fastest) a count of arrivals followed by one
        line per arrival::

            amplitude  phase_deg  delay_s  delay_imag_s  src_angle  rcv_angle  n_top  n_bottom

        Returns
        -------
        list[list[Eigenray]]
            The arrivals at each grid point, in file order.

        """
        receivers: list[list[Eigenray]] = []
        with open(arr_path) as f:
            f.readline()  # '2D'
            f.readline()  # frequency
            num_sources = int(f.readline().split()[0])
            num_receiver_depths = int(f.readline().split()[0])
            num_ranges = int(f.readline().split()[0])
            f.readline()  # maximum number of arrivals, a pre-allocation hint
            for _ in range(num_sources * num_receiver_depths * num_ranges):
                arrivals = []
                for _ in range(int(f.readline())):
                    parts = f.readline().split()
                    arrivals.append(
                        Eigenray(
                            amplitude=float(parts[0]),
                            phase_rad=float(np.radians(float(parts[1]))),
                            delay_s=float(parts[2]),
                            src_angle_deg=float(parts[4]),
                            rcv_angle_deg=float(parts[5]),
                            n_surface_bounces=int(parts[6]),
                            n_bottom_bounces=int(parts[7]),
                        )
                    )
                receivers.append(arrivals)
        return receivers


def _transfer_function(rays: list[Eigenray], frequencies_hz: FloatArray) -> ComplexArray:
    r"""Sum arrivals into a transfer function over ``frequencies_hz``.

    The channel is real, so negative frequencies get the complex conjugate of the positive
    ones, :math:`H(-f) = H(f)^*`, and :math:`H(0)` is real.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    transfer_function = np.zeros(frequencies_hz.size, dtype=np.complex128)
    if not rays:
        return transfer_function
    amplitudes = np.array([r.amplitude for r in rays])
    phases_rad = np.array([r.phase_rad for r in rays])
    delays_s = np.array([r.delay_s for r in rays])
    weights = amplitudes * np.exp(1j * phases_rad)
    positive, negative = frequencies_hz > 0, frequencies_hz < 0
    if np.any(positive):
        transfer_function[positive] = _delay_sum(weights, delays_s, frequencies_hz[positive])
    if np.any(negative):
        transfer_function[negative] = np.conj(
            _delay_sum(weights, delays_s, -frequencies_hz[negative])
        )
    # At f = 0 only the real part of each arrival's phase term survives.
    transfer_function[frequencies_hz == 0] = amplitudes @ np.cos(phases_rad)
    return transfer_function


#: Bellhop's bathymetry interpolation codes, by the name of the setting.
_BATHYMETRY_INTERPOLATIONS = {"curvilinear": "C", "linear": "L"}

#: Format of every depth written to Bellhop's files, in metres: to the micrometre. Coarser
#: rounding puts kinks into sloped bathymetry (see ``_write_bty``).
_DEPTH_FORMAT = ".6f"

#: Below this many frequencies the direct sum is as quick as the factored one.
_MIN_FACTORED_FREQUENCIES = 64
#: Arrivals summed at once in the factored sum, which bounds its memory.
_ARRIVAL_CHUNK = 16


def _delay_sum(
    weights: ComplexArray, delays_s: FloatArray, frequencies_hz: FloatArray
) -> ComplexArray:
    r"""Return :math:`\sum_k w_k e^{-i 2 \pi f \tau_k}` at each frequency.

    FFT grids put every frequency on a lattice, :math:`f = f_0 + j \Delta f` for whole
    numbers :math:`j`, perhaps with gaps where bins were left out. Write :math:`j = qB + p`:
    each term factors into :math:`e^{-i 2 \pi \tau_k (f_0 + qB \Delta f)}
    e^{-i 2 \pi \tau_k p \Delta f}`, so the sum over arrivals over the whole lattice is a
    matrix product of two small tables, about :math:`2\sqrt{N}` exponentials per arrival
    rather than :math:`N`, and the requested frequencies are picked from it. The result is the
    same to rounding. Grids that aren't on a lattice, or are too short or sparse for it to pay,
    are summed directly.
    """
    lattice = _lattice_indices(frequencies_hz)
    if lattice is None:
        return weights @ np.exp(-2j * np.pi * np.outer(delays_s, frequencies_hz))
    start_hz, step_hz, indices = lattice
    size = int(indices.max()) + 1
    block = int(np.ceil(np.sqrt(size)))
    num_blocks = int(np.ceil(size / block))
    within = np.exp(-2j * np.pi * np.outer(delays_s, step_hz * np.arange(block)))
    starts_hz = start_hz + step_hz * block * np.arange(num_blocks)
    across = np.exp(-2j * np.pi * np.outer(starts_hz, delays_s)) * weights[None, :]
    # The product of the two tables, summed over arrivals by broadcasting rather than matmul:
    # OpenBLAS's threading makes a product this small tens of times slower. Arrivals are
    # taken in chunks to bound the memory of the broadcast.
    table = np.zeros((num_blocks, block), dtype=np.complex128)
    for first in range(0, delays_s.size, _ARRIVAL_CHUNK):
        chunk = slice(first, first + _ARRIVAL_CHUNK)
        table += np.sum(across[:, chunk, None] * within[None, chunk, :], axis=1)
    return table.reshape(-1)[indices]


def _lattice_indices(frequencies_hz: FloatArray) -> tuple[float, float, NDArray[np.intp]] | None:
    """Return ``(f_0, step, indices)`` with ``f = f_0 + indices * step``, or None.

    None when there are too few frequencies for the factored sum to pay, when they aren't on a
    common lattice, or when the lattice they span is more than four times their number.
    """
    if frequencies_hz.size < _MIN_FACTORED_FREQUENCIES:
        return None
    start_hz = float(frequencies_hz.min())
    steps = np.diff(np.unique(frequencies_hz))
    if steps.size == 0:
        return None
    # Refine the step over the whole span: a single difference carries rounding error, which
    # would grow with the index.
    span_hz = float(frequencies_hz.max()) - start_hz
    step_hz = span_hz / round(span_hz / float(steps.min()))
    position = (frequencies_hz - start_hz) / step_hz
    indices = np.rint(position).astype(np.intp)
    if not np.allclose(position, indices, rtol=0.0, atol=1e-6):
        return None
    if indices.max() + 1 > 4 * frequencies_hz.size:
        return None
    return start_hz, step_hz, indices

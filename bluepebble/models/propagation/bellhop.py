"""Eigenray propagation with Bellhop (``bellhopcxx``) in arrivals mode.

Bellhop is an external program, not a Python dependency: install ``bellhopcxx`` from the
bellhopcuda project (https://github.com/A-New-BellHope/bellhopcuda) and put it on ``PATH``, or
give its full path as ``exe_path``.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property

from ..environment import Bathymetry
from .absorption import thorp_db_per_km
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

    from ...arrays import PlatformState

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]

_RECEIVER_GEOMETRIES = ("reference", "per_sensor")
_VOLUME_ATTENUATIONS = ("none", "thorp")
# The dB/km in one Np/m, as bellhopcxx rounds it, so its imaginary delays invert exactly.
_DB_PER_KM_PER_NP_PER_M = 8685.8896


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
        It is geometric: spreading and boundary losses, without seawater absorption.
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
    path_length_m : float or None
        Length of the path in metres, or None if the model ran with
        ``volume_attenuation="none"``, when Bellhop does not report it.

    """

    amplitude: float
    phase_rad: float
    delay_s: float
    src_angle_deg: float
    rcv_angle_deg: float
    n_surface_bounces: int
    n_bottom_bounces: int
    path_length_m: float | None = None


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
    :math:`n` with arrivals :math:`k` of amplitude :math:`a_k`, phase :math:`\phi_k`, delay
    :math:`\tau_k` and path length :math:`L_k`,

    .. math::

        H_n(f) = \sum_k a_k \, 10^{-\alpha(f) L_k / 20} \, e^{i \phi_k} \,
            e^{-i 2 \pi f \tau_k}, \qquad f > 0

    and :math:`H_n(-f) = H_n(f)^*` (a real channel). This is Blue Pebble's convention, in which
    a delay :math:`\tau` is :math:`e^{-i 2 \pi f \tau}`, and it is also the convention of
    Bellhop's reported phases: a bottom reflection's phase equals the phase of the Rayleigh
    reflection coefficient in it. The seawater absorption :math:`\alpha`, in dB/km with
    :math:`L_k` in km, is zero with ``volume_attenuation="none"``, and
    :func:`~bluepebble.models.propagation.thorp_db_per_km` with ``"thorp"``.

    Absorption is applied here, to each arrival at each frequency, and not by Bellhop. Bellhop
    evaluates its own volume attenuation (a fourth letter of the top option, ``'T'`` for
    Thorp) at the run frequency :math:`f_0` only. In arrivals mode it leaves the amplitudes
    alone and reports the absorption as an imaginary part of each delay,
    :math:`2 \pi f_0 \operatorname{Im} \tau_k = -\alpha(f_0) L_k` with :math:`\alpha` in Np/m.
    With ``"thorp"`` the model runs Bellhop with that option only to read :math:`L_k` from it.

    Parameters
    ----------
    ssp : SoundSpeedProfile
        Sound speed profile, used as a function of depth only.
    bathymetry : Bathymetry
        Water depth, sampled along each run's vertical plane.
    receiver_geometry : str, optional
        ``"reference"`` makes one run in the vertical plane from the source towards the array
        reference point, with every sensor at its own horizontal range and depth.
        ``"per_sensor"`` makes one run per sensor, each in the plane through that sensor.
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
    volume_attenuation : str, optional
        Seawater absorption: ``"none"`` or ``"thorp"``. Thorp only: ``bellhopcxx``'s
        Francois–Garrison (``'F'``) computes the boric-acid relaxation frequency with
        ``pow(1.0, ...)`` instead of ``10 ** (...)``, so it is wrong at every frequency (×0.5
        to ×1.3 over 0.5–10 kHz) and cannot serve as a reference.

    Assumptions
    -----------
    - Propagation is in a vertical plane through the source (2D; Nx2D is not used), so
      out-of-plane bathymetry and refraction are ignored. With ``"reference"`` geometry the
      whole array shares the plane towards its reference point, which is exact when the
      bathymetry is the same on every bearing and close when the array's aperture is small
      against the range.
    - The sound speed varies with depth only: the same profile applies at every range.
    - The surface is a flat pressure-release (vacuum) boundary.
    - The bottom is a fluid half-space (no shear) below the bathymetry, with a flat-interface
      reflection coefficient and no roughness.
    - In :meth:`propagate_spectrum`, Bellhop runs once at the centre of the requested band, so
      each arrival's amplitude and phase (spreading and bottom loss) are taken as constant
      across the band; only the delay phase and the absorption vary with frequency.
    - With ``"thorp"``, absorption depends on frequency only, so it is the same all along a
      path whatever the temperature, salinity, depth or pH.
    - With ``"thorp"``, Bellhop also adds the Thorp term to the sediment half-space's
      attenuation at the run frequency. At 10 km range in 4 km of isovelocity water over the
      default bottom, at 0.5 to 10 kHz, that changed bottom-reflected amplitudes by less than
      1e-6 of their value and their phases by less than 0.002 degrees.

    """

    bathymetry: Bathymetry = Property(doc="Bathymetry model (water depth)")
    receiver_geometry: str = Property(
        default="reference",
        doc='"reference" (one run towards the array reference point) or "per_sensor" (one '
        "run per sensor)",
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
    volume_attenuation: Literal["none", "thorp"] = Property(
        default="none",
        doc='Seawater absorption: "none" or "thorp" (thorp_db_per_km, applied to each arrival '
        "at each frequency). Thorp only: bellhopcxx's Francois-Garrison ('F') computes the "
        "boric-acid relaxation frequency with pow(1.0, ...) instead of 10 ** (...), so it is "
        "wrong at every frequency (x0.5 to x1.3 over 0.5-10 kHz) and cannot serve as a "
        "reference.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the model and check ``receiver_geometry`` and ``volume_attenuation``.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        Raises
        ------
        ValueError
            If ``receiver_geometry`` is not ``"reference"`` or ``"per_sensor"``, or
            ``volume_attenuation`` is not ``"none"`` or ``"thorp"``.

        """
        super().__init__(*args, **kwargs)
        if self.receiver_geometry not in _RECEIVER_GEOMETRIES:
            raise ValueError(
                f"receiver_geometry ({self.receiver_geometry!r}) must be one of "
                f"{_RECEIVER_GEOMETRIES}."
            )
        if self.volume_attenuation not in _VOLUME_ATTENUATIONS:
            raise ValueError(
                f"volume_attenuation ({self.volume_attenuation!r}) must be one of "
                f"{_VOLUME_ATTENUATIONS}."
            )

    def compute_eigenrays(
        self,
        source_position: ArrayLike,
        receiver_position: ArrayLike,
        frequency_hz: float,
    ) -> list[Eigenray]:
        """Run Bellhop for one source and receiver and return every arrival.

        The amplitudes are geometric whatever ``volume_attenuation`` is: absorption is applied
        only in :meth:`propagate_spectrum` and :meth:`propagate`. With ``"thorp"`` each arrival
        also carries its ``path_length_m``.

        Parameters
        ----------
        source_position : ArrayLike
            World-frame ``(x, y, z)`` of the source in metres, with z negative below the
            surface (Blue Pebble's +z up convention).
        receiver_position : ArrayLike
            World-frame ``(x, y, z)`` of the receiver in metres, with the same convention.
        frequency_hz : float
            Frequency of the run in Hz. With a fixed ``num_beams`` the arrivals do not depend
            on it, because the beams are geometric and the bottom attenuation is per
            wavelength. With ``num_beams=0`` Bellhop sizes the fan from it.

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
        arrivals = self._sensor_arrivals(platform, source, band_centre_hz)
        transfer_functions = np.stack(
            [self._absorbed_transfer_function(rays, frequencies) for rays in arrivals]
        )
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
            magnitude = abs(self._absorbed_transfer_function(rays, np.array([frequency_hz]))[0])
            losses_db.append(999.0 if magnitude < 1e-12 else float(-20 * np.log10(magnitude)))
            if np.isnan(travel_time_s):
                travel_time_s = min((r.delay_s for r in rays), default=np.nan)
        loss: PropagationLoss = (
            np.asarray(losses_db, dtype=np.float64) if self.use_all_frequencies else losses_db[0]
        )
        return loss, float(travel_time_s)

    def _arrival_absorption(self, rays: list[Eigenray], frequencies_hz: ArrayLike) -> FloatArray:
        r"""Return the factor absorption scales each arrival's amplitude by at each frequency.

        The factor is :math:`10^{-\alpha(|f|) L_k / 20}`, with :math:`\alpha` in dB/km from
        :func:`~bluepebble.models.propagation.thorp_db_per_km` and :math:`L_k` the arrival's
        ``path_length_m`` in km.

        Parameters
        ----------
        rays : list[Eigenray]
            Arrivals from a run of this model.
        frequencies_hz : ArrayLike
            Frequencies in Hz.

        Returns
        -------
        FloatArray
            Shape ``(num_arrivals, num_frequencies)``; all ones with
            ``volume_attenuation="none"``.

        Raises
        ------
        ValueError
            If absorption is on and an arrival has no ``path_length_m``, so it came from a
            run without it.

        """
        frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
        if self.volume_attenuation == "none":
            return np.ones((len(rays), frequencies.size))
        if any(r.path_length_m is None for r in rays):
            raise ValueError(
                "An arrival has no path_length_m: it came from a run with "
                'volume_attenuation="none".'
            )
        lengths_km = np.array([r.path_length_m for r in rays], dtype=float) / 1000.0
        return 10.0 ** (-np.outer(lengths_km, thorp_db_per_km(frequencies)) / 20.0)

    def _absorbed_transfer_function(
        self, rays: list[Eigenray], frequencies_hz: FloatArray
    ) -> ComplexArray:
        """Sum ``rays`` into a transfer function, with absorption per ``volume_attenuation``."""
        if self.volume_attenuation == "none":
            return _transfer_function(rays, frequencies_hz)
        return _transfer_function(
            rays, frequencies_hz, absorption=self._arrival_absorption(rays, frequencies_hz)
        )

    def _imag_delay_to_path_length_mps(self, frequency_hz: float) -> float | None:
        r"""Return the ratio of an arrival's path length to its imaginary delay, in m/s.

        Bellhop integrates the delay along the path with a complex sound speed, so
        :math:`2 \pi f_0 \operatorname{Im} \tau = -\alpha(f_0) L` with :math:`\alpha` in Np/m
        at the run frequency :math:`f_0`. None with ``volume_attenuation="none"``, when the
        delay is real.
        """
        if self.volume_attenuation == "none":
            return None
        alpha_np_per_m = float(thorp_db_per_km(frequency_hz)) / _DB_PER_KM_PER_NP_PER_M
        return -2 * np.pi * frequency_hz / alpha_np_per_m

    def _sensor_arrivals(
        self, platform: "PlatformState", source: "State", frequency_hz: float
    ) -> list[list[Eigenray]]:
        """Return the arrivals at every sensor, per ``receiver_geometry``."""
        source_position = _get_source_position(source)
        sensors = np.asarray(platform.array.state_vector, dtype=float)
        if self.receiver_geometry == "reference":
            reference_xy = np.asarray(platform.array.ref_state_vector, dtype=float).ravel()[:2]
            return self._run_arrivals(source_position, sensors, reference_xy, frequency_hz)
        return [
            self._run_arrivals(
                source_position, sensors[:, [n]], sensors[:2, n], frequency_hz
            )[0]
            for n in range(sensors.shape[1])
        ]

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
            self._write_bty(stem, bathy_range_km, bottom_depth_m)
            self._run_bellhopcxx(exe, stem, self.memory)
            blocks = self._parse_arr(
                stem.with_suffix(".arr"),
                imag_delay_to_path_length_mps=self._imag_delay_to_path_length_mps(frequency_hz),
            )
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
        depths = np.linspace(0.0, water_depth_m, self.n_ssp_samples)
        speeds = [float(self.ssp.calculate(d)) for d in depths]
        ssp_lines = "\n".join(
            f"  {d:.2f}  {c:.4f}  /" for d, c in zip(depths, speeds, strict=True)
        )
        receiver_depths = np.atleast_1d(receiver_depths_m)
        ranges_km = np.atleast_1d(receiver_ranges_m) / 1000.0
        low_angle, high_angle = self.beam_angle_limits_deg
        # Spline SSP, vacuum above the surface, attenuation in dB/wavelength, then Thorp
        # volume attenuation if asked for.
        top_option = "SVWT" if self.volume_attenuation == "thorp" else "SVW"
        env = "\n".join(
            [
                "'BellhopArrivals'",
                f"{frequency_hz:.4f}",
                "1",  # number of media
                f"'{top_option}'",
                f"0 0.0 {water_depth_m:.2f}",
                ssp_lines,
                "'A~' 0.0",  # acousto-elastic half-space below a .bty bathymetry, no roughness
                f"{water_depth_m:.2f}  {self.bottom_sound_speed_mps}  0.0  "
                f"{self.bottom_density_g_cm3}  {self.bottom_attenuation_db_per_wavelength}  /",
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

    @staticmethod
    def _write_bty(stem: Path, range_km: FloatArray, depth_m: FloatArray) -> None:
        """Write a piecewise-linear range-dependent bathymetry file."""
        lines = ["'L'", str(len(range_km))]
        lines += [f"{r:.6f} {d:.2f}" for r, d in zip(range_km, depth_m, strict=True)]
        stem.with_suffix(".bty").write_text("\n".join(lines))

    @staticmethod
    def _run_bellhopcxx(exe: str, stem: Path, memory: str) -> None:
        """Run ``bellhopcxx`` on ``stem``, raising with its output if it fails."""
        try:
            subprocess.run(
                [exe, f"--memory={memory}", str(stem)], capture_output=True, check=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"bellhopcxx failed (exit {e.returncode}).\n"
                f"env file: {stem}.env\nstdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e

    @staticmethod
    def _parse_arr(
        arr_path: Path, *, imag_delay_to_path_length_mps: float | None = None
    ) -> list[list[Eigenray]]:
        """Parse a 2D ASCII arrivals file for a rectilinear receiver grid.

        The file holds a header (``'2D'``, the frequency, the source depths, receiver depths
        and ranges, each count first, and a maximum-arrivals hint), then for every source,
        receiver depth and range (range varying fastest) a count of arrivals followed by one
        line per arrival::

            amplitude  phase_deg  delay_s  delay_imag_s  src_angle  rcv_angle  n_top  n_bottom

        Parameters
        ----------
        arr_path : Path
            The arrivals file.
        imag_delay_to_path_length_mps : float, optional
            If given, each arrival's ``path_length_m`` is this times its imaginary delay.

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
                            path_length_m=(
                                None
                                if imag_delay_to_path_length_mps is None
                                else imag_delay_to_path_length_mps * float(parts[3])
                            ),
                        )
                    )
                receivers.append(arrivals)
        return receivers


def _transfer_function(
    rays: list[Eigenray], frequencies_hz: FloatArray, *, absorption: FloatArray | None = None
) -> ComplexArray:
    r"""Sum arrivals into a transfer function over ``frequencies_hz``.

    The channel is real, so negative frequencies get the complex conjugate of the positive
    ones, :math:`H(-f) = H(f)^*`, and :math:`H(0)` is real. ``absorption``, of shape
    ``(num_arrivals, num_frequencies)``, scales each arrival's amplitude at each frequency.
    """
    if not rays:
        return np.zeros(frequencies_hz.size, dtype=np.complex128)
    amplitudes = np.array([r.amplitude for r in rays])
    phases_rad = np.array([r.phase_rad for r in rays])
    delays_s = np.array([r.delay_s for r in rays])
    sign = np.sign(frequencies_hz)
    # Arrival phase rotates with sign(f); at f = 0 only its real part survives.
    arrival_terms = np.where(
        sign[None, :] == 0,
        np.cos(phases_rad)[:, None],
        np.exp(1j * sign[None, :] * phases_rad[:, None]),
    )
    delay_terms = np.exp(-2j * np.pi * delays_s[:, None] * frequencies_hz[None, :])
    if absorption is None:
        return np.asarray(amplitudes @ (arrival_terms * delay_terms), dtype=np.complex128)
    return np.asarray(
        np.sum(amplitudes[:, None] * absorption * arrival_terms * delay_terms, axis=0),
        dtype=np.complex128,
    )

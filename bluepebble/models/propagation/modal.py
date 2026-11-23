"""Normal-mode propagation in a range-independent waveguide, from KRAKEN or KRAKENC."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from stonesoup.base import Property

from ..environment import Bathymetry, TabulatedSoundSpeedProfile
from ..environment.seabed import LayeredSeabed
from .acoustic import (
    AcousticPropagationModel,
    FloatArray,
    SpectrumPropagationModel,
    SpectrumResult,
    _get_source_position,
)
from .kraken import AcousticsToolboxBinaries, KrakenEnvironment, ModeSet, compute_mode_set

if TYPE_CHECKING:
    from stonesoup.types.state import State

    from ...arrays import PlatformState


@dataclass(frozen=True)
class ModalDispersion:
    """Spread of modal arrival times, used to place a waveguide's arrivals in a filter.

    For a receiver at range ``r``, the modes that matter arrive between
    ``r / max_group_speed_m_s`` and ``r / min_group_speed_m_s``. Delaying by
    ``r / reference_speed_m_s`` centres that window, leaving at most
    ``r * half_spread_s_per_m`` either side.

    Attributes
    ----------
    reference_speed_m_s : float
        Speed whose delay centres the modal arrivals.
    min_group_speed_m_s, max_group_speed_m_s : float
        Group-speed range of the significant modes.
    max_residual_wavenumber : float
        Largest ``|Re k_m - omega / c_ref|`` among the significant modes, in rad/m: how fast
        a mode's phase turns, relative to the bulk delay, as range changes.

    """

    reference_speed_m_s: float
    min_group_speed_m_s: float
    max_group_speed_m_s: float
    max_residual_wavenumber: float

    @property
    def half_spread_s_per_m(self) -> float:
        """Half the arrival-time spread per metre of range, in s/m."""
        return 0.5 * (1.0 / self.min_group_speed_m_s - 1.0 / self.max_group_speed_m_s)


class KrakenModalPropagationModel(AcousticPropagationModel, SpectrumPropagationModel):
    """Normal-mode transfer functions for a range-independent waveguide.

    The modes depend only on the environment and frequency, not on where the source and
    sensors are, so KRAKENC (or KRAKEN) runs once for a frequency grid and depth span
    and every later call is a mode sum at the current ranges and depths. The sum
    reproduces FIELD's point-source pressure (see :mod:`.kraken.modes`): relative to 1 m,
    with the ``exp(-2j pi f tau)`` delay convention used by the other models.

    Sensor and source depths may be given with either sign. Ranges are horizontal.

    The transfer function is the far-field mode sum: it omits the continuous spectrum, so
    it is accurate beyond a water depth or two from the source, and holds only where
    the waveguide really is range independent.
    """

    bathymetry: Bathymetry = Property(
        doc="Flat bathymetry giving the water depth; must not vary with position"
    )
    seabed: LayeredSeabed = Property(doc="Sediment layers and halfspace beneath the water")
    thorp_attenuation: bool = Property(default=True, doc="Add Thorp volume attenuation")
    program: str = Property(default="krakenc", doc="Mode solver: 'krakenc' or 'kraken'")
    c_low_m_s: float = Property(default=0.0, doc="Lowest phase speed searched; 0 lets KRAKEN pick")
    c_high_m_s: float | None = Property(
        default=None,
        doc="Highest phase speed kept; defaults to the halfspace speed (trapped modes only)",
    )
    depth_range_m: tuple[float, float] | None = Property(
        default=None,
        doc="Depth span to tabulate modes over; by default the depths requested plus "
        "depth_margin_m",
    )
    depth_margin_m: float = Property(
        default=10.0, doc="Margin added around requested depths when choosing the span"
    )
    depth_spacing_m: float | None = Property(
        default=None,
        doc="Tabulation spacing; defaults to a tenth of the shortest wavelength in water",
    )
    rmax_km: float = Property(
        default=20.0,
        doc="Range KRAKEN converges its mesh for, in km; a simulator raises it to the "
        "furthest range it needs. Unused by KRAKENC, which solves on one fine mesh",
    )
    mesh_points_per_wavelength: float | None = Property(
        default=None,
        doc="Finite-difference mesh density; by default 80 for KRAKENC and KRAKEN's own "
        "choice, refined by Richardson extrapolation, for KRAKEN",
    )
    ssp_resolution_m: float = Property(
        default=1.0, doc="Sampling of an analytic SSP for KRAKEN, in metres"
    )
    min_range_m: float = Property(
        default=1.0, doc="Ranges below this are clamped to it, with a warning"
    )
    nominal_frequency_hz: float = Property(
        default=1000.0, doc="Frequency used by propagate() for transmission loss"
    )
    dispersion_threshold_db: float = Property(
        default=40.0,
        doc="Modes weaker than this below the strongest are ignored when measuring dispersion",
    )
    num_workers: int | None = Property(default=None, doc="Parallel KRAKEN runs")
    frequencies_per_run: int = Property(default=16, doc="Frequencies per broadband KRAKEN run")
    cache_dir: str | None = Property(default=None, doc="Directory caching computed mode sets")
    binary_search_paths: Sequence[str] = Property(
        default=(), doc="Directories searched for the toolbox programs before PATH"
    )
    mode_set: ModeSet | None = Property(
        default=None, doc="Precomputed modes; KRAKEN is not run if they cover a request"
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Validate the environment."""
        super().__init__(*args, **kwargs)
        depth = abs(float(self.bathymetry.get_depth(0.0, 0.0)))
        for x, y in ((1e5, 0.0), (0.0, 1e5), (-1e5, -1e5)):
            if not np.isclose(abs(float(self.bathymetry.get_depth(x, y))), depth):
                raise ValueError(
                    "KrakenModalPropagationModel needs a flat (range-independent) bathymetry"
                )
        if self.program not in {"krakenc", "kraken"}:
            raise ValueError(f"program must be 'krakenc' or 'kraken', got {self.program!r}")
        if self.seabed.is_elastic and self.program != "krakenc":
            raise ValueError("Elastic seabeds need program='krakenc'")
        self._water_depth_m = depth
        self._mode_sets: list[ModeSet] = [self.mode_set] if self.mode_set is not None else []

    # ----------------------------------------------------------------------------------
    # Environment and modes
    # ----------------------------------------------------------------------------------
    @property
    def water_depth_m(self) -> float:
        """Water depth in metres."""
        return self._water_depth_m

    def kraken_environment(self) -> KrakenEnvironment:
        """Describe the waveguide for KRAKEN, with the SSP sampled from 0 m to the seabed."""
        depth = self.water_depth_m
        if isinstance(self.ssp, TabulatedSoundSpeedProfile):
            table = np.asarray(self.ssp._depths, dtype=float)
            inner = table[(table > 0.0) & (table < depth)]
            z = np.concatenate(([0.0], inner, [depth]))
        else:
            count = max(2, int(np.ceil(depth / float(self.ssp_resolution_m))) + 1)
            z = np.linspace(0.0, depth, count)
        c = np.asarray(self.ssp.calculate(z), dtype=float).reshape(-1)
        if c.size == 1:
            c = np.full(z.size, float(c[0]))
        return KrakenEnvironment(
            ssp_depths_m=tuple(float(v) for v in z),
            ssp_speeds_m_s=tuple(float(v) for v in c),
            water_depth_m=depth,
            seabed=self.seabed,
            thorp_attenuation=bool(self.thorp_attenuation),
        )

    def default_depth_spacing_m(self, frequencies_hz: ArrayLike) -> float:
        """Return a tenth of the shortest wavelength in water at the highest frequency."""
        if self.depth_spacing_m is not None:
            return float(self.depth_spacing_m)
        c_min = self.kraken_environment().min_speed_m_s
        return c_min / float(np.max(np.asarray(frequencies_hz, dtype=float))) / 10.0

    def modes_for(
        self,
        frequencies_hz: ArrayLike,
        depths_m: ArrayLike = (),
        rmax_km: float | None = None,
    ) -> ModeSet:
        """Return modes covering some frequencies and depths, computing them if needed.

        Parameters
        ----------
        frequencies_hz : array-like
            Frequencies the modes must be on, in Hz.
        depths_m : array-like, optional
            Depths that must lie inside the tabulation; ignored if ``depth_range_m`` is set.
        rmax_km : float, optional
            Furthest range of interest; the larger of this and ``rmax_km`` is used.

        Returns
        -------
        ModeSet
            A mode set on exactly these frequencies.

        """
        frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
        depths = np.abs(np.asarray(depths_m, dtype=float).reshape(-1))
        for modes in self._mode_sets:
            if self._covers(modes, frequencies, depths):
                return modes

        if self.depth_range_m is not None:
            top, bottom = sorted(abs(float(v)) for v in self.depth_range_m)
        elif depths.size:
            margin = float(self.depth_margin_m)
            top, bottom = depths.min() - margin, depths.max() + margin
        else:
            top, bottom = 0.0, self.water_depth_m
        top, bottom = max(top, 0.0), min(bottom, self.water_depth_m)
        if depths.size and (depths.max() > self.water_depth_m):
            raise ValueError("Depths below the seabed are not supported by the simulator")
        spacing = self.default_depth_spacing_m(frequencies)
        count = max(2, int(np.ceil((bottom - top) / spacing)) + 1)
        table = np.linspace(top, bottom, count)
        modes = compute_mode_set(
            self.kraken_environment(),
            frequencies,
            table,
            program=self.program,
            binaries=AcousticsToolboxBinaries.find(self.binary_search_paths),
            c_low_m_s=float(self.c_low_m_s),
            c_high_m_s=self.c_high_m_s,
            rmax_km=max(float(self.rmax_km), float(rmax_km or 0.0)),
            num_workers=self.num_workers,
            frequencies_per_run=int(self.frequencies_per_run),
            cache_dir=self.cache_dir,
            mesh_points_per_wavelength=self.mesh_points_per_wavelength,
        )
        self._mode_sets.append(modes)
        return modes

    @staticmethod
    def _covers(modes: ModeSet, frequencies: FloatArray, depths: FloatArray) -> bool:
        if modes.frequencies_hz.shape != frequencies.shape or not np.allclose(
            modes.frequencies_hz, frequencies, rtol=1e-9, atol=0.0
        ):
            return False
        if depths.size == 0:
            return True
        tolerance = 1e-6 * max(float(modes.depths_m[-1] - modes.depths_m[0]), 1.0)
        return bool(
            depths.min() >= modes.depths_m[0] - tolerance
            and depths.max() <= modes.depths_m[-1] + tolerance
        )

    def dispersion(self, modes: ModeSet, source_depth_m: float, range_m: float) -> ModalDispersion:
        """Measure the spread of modal arrivals from a source depth at a range.

        Only modes within ``dispersion_threshold_db`` of the strongest, after attenuation
        over ``range_m``, count; heavily attenuated steep modes would otherwise set the
        spread.

        Parameters
        ----------
        modes : ModeSet
            The modes.
        source_depth_m : float
            Source depth in metres.
        range_m : float
            Range in metres, usually the furthest the source will be.

        Returns
        -------
        ModalDispersion
            Group-speed range and the speed that centres it.

        """
        significant = modes.significant_modes(
            source_depth_m, max(range_m, 1.0), float(self.dispersion_threshold_db)
        )
        speeds = modes.group_speeds()
        valid = significant & np.isfinite(speeds)
        # Clip finite-difference outliers to physically possible speeds.
        c_water = self.kraken_environment().min_speed_m_s
        c_max = max(self.seabed.max_speed_m_s, c_water)
        if np.any(valid):
            values = np.clip(speeds[valid], 0.5 * c_water, c_max)
            v_min, v_max = float(values.min()), float(values.max())
        else:
            phase = 2.0 * np.pi * modes.frequencies_hz[:, None] / modes.k.real
            values = phase[significant] if np.any(significant) else np.array([c_water])
            v_min, v_max = float(values.min()), float(values.max())
        c_ref = 2.0 / (1.0 / v_min + 1.0 / v_max)
        omega = 2.0 * np.pi * modes.frequencies_hz[:, None]
        residual = np.abs(modes.k.real - omega / c_ref)
        use = significant if np.any(significant) else modes.mode_mask
        max_residual = float(residual[use].max()) if np.any(use) else 0.0
        return ModalDispersion(
            reference_speed_m_s=c_ref,
            min_group_speed_m_s=v_min,
            max_group_speed_m_s=v_max,
            max_residual_wavenumber=max_residual,
        )

    # ----------------------------------------------------------------------------------
    # Transfer functions
    # ----------------------------------------------------------------------------------
    def geometry(
        self, platform: PlatformState, source: State
    ) -> tuple[FloatArray, FloatArray, float]:
        """Horizontal ranges and depths of each sensor, and the source depth.

        Parameters
        ----------
        platform : PlatformState
            Platform state providing ``array.state_vector``.
        source : State
            Source state with ``position_mapping`` metadata.

        Returns
        -------
        ranges_m, sensor_depths_m : numpy.ndarray
            Per sensor, clamped to ``min_range_m``.
        source_depth_m : float
            Source depth.

        """
        source_position = _get_source_position(source).reshape(-1)
        sensors = np.asarray(platform.array.state_vector, dtype=float)
        return self.geometry_from_positions(sensors, source_position)

    def geometry_from_positions(
        self, sensors: FloatArray, source_position: FloatArray
    ) -> tuple[FloatArray, FloatArray, float]:
        """As :meth:`geometry`, from a ``(3, S)`` sensor array and a source position."""
        offsets = sensors[:2] - source_position[:2, np.newaxis]
        ranges = np.hypot(offsets[0], offsets[1])
        if np.any(ranges < self.min_range_m):
            warnings.warn(
                f"Ranges below {self.min_range_m} m are clamped; the mode sum is a far-field "
                "approximation",
                stacklevel=3,
            )
            ranges = np.maximum(ranges, float(self.min_range_m))
        return ranges, np.abs(sensors[2]), abs(float(source_position[2]))

    def propagate_spectrum(
        self, platform: PlatformState, source: State, frequencies_hz: ArrayLike
    ) -> SpectrumResult:
        """Per-sensor transfer functions from a mode sum.

        Parameters
        ----------
        platform : PlatformState
            Platform state providing sensor positions.
        source : State
            Source state.
        frequencies_hz : array-like
            Frequencies in Hz. Modes are computed for this exact grid on first use.

        Returns
        -------
        SpectrumResult
            ``(H, delays_s)``: ``H`` has shape ``(num_sensors, num_frequencies)``, and each
            sensor's delay is its range over the speed that centres the modal arrivals.

        """
        ranges, depths, source_depth = self.geometry(platform, source)
        modes = self.modes_for(
            frequencies_hz, np.concatenate((depths, [source_depth])), rmax_km=ranges.max() / 1e3
        )
        transfer = modes.transfer(source_depth, depths, ranges)
        c_ref = self.dispersion(modes, source_depth, float(ranges.max())).reference_speed_m_s
        return transfer, ranges / c_ref

    def propagate(self, platform: PlatformState, source: State) -> tuple[float, float]:
        """Transmission loss to the reference sensor at ``nominal_frequency_hz``.

        Parameters
        ----------
        platform : PlatformState
            Platform state.
        source : State
            Source state.

        Returns
        -------
        tuple of float
            Transmission loss in dB and the delay that centres the modal arrivals, in s.

        """
        source_position = _get_source_position(source).reshape(-1)
        reference = np.asarray(platform.array.ref_state_vector, dtype=float).reshape(3, 1)
        ranges, depths, source_depth = self.geometry_from_positions(reference, source_position)
        frequency = np.array([float(self.nominal_frequency_hz)])
        modes = self.modes_for(
            frequency, np.concatenate((depths, [source_depth])), rmax_km=ranges.max() / 1e3
        )
        pressure = modes.transfer(source_depth, depths, ranges)[0, 0]
        c_ref = self.dispersion(modes, source_depth, float(ranges[0])).reference_speed_m_s
        loss = -20.0 * np.log10(max(abs(pressure), 1e-300))
        return float(loss), float(ranges[0] / c_ref)

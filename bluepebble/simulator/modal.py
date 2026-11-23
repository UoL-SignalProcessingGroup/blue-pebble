"""Broadband passive sonar simulation in a range-independent normal-mode waveguide."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property

from ..models.propagation.acoustic import _get_source_position
from ..models.propagation.kraken import ModeSet
from ..models.propagation.modal import KrakenModalPropagationModel, ModalDispersion
from ..signal.base import SourceSignal
from .continuous import ChannelHistory, ContinuousChannelFilterPassiveSonarArraySimulator

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complexfloating[Any, Any]]


@dataclass
class ModalKnots:
    """Where and when one source's channel is evaluated.

    Attributes
    ----------
    knot_times_s : numpy.ndarray
        Knot times from ``start_time``, shape ``(K,)``.
    ranges_m, sensor_depths_m : numpy.ndarray
        Horizontal range and depth of each sensor at each knot, shape ``(K, S)``.
    source_depths_m : numpy.ndarray
        Source depth at each knot, shape ``(K,)``.
    active : numpy.ndarray
        Whether the source is sounding at each knot, shape ``(K,)``.
    reference_delay_s : numpy.ndarray
        Bulk delay at each knot: the median sensor range over ``bulk_speed_m_s``.
    dispersion : ModalDispersion
        The modal arrival spread the knots were planned for.
    bulk_speed_m_s : float
        Speed the bulk delay uses. Where the filter has room it is nearer the modes' phase
        speeds than ``dispersion.reference_speed_m_s``, so fewer knots are needed.
    max_residual_wavenumber : float
        Largest ``|Re k_m - omega / bulk_speed_m_s|`` among the significant modes, rad/m.

    """

    knot_times_s: FloatArray
    ranges_m: FloatArray
    sensor_depths_m: FloatArray
    source_depths_m: FloatArray
    active: NDArray[np.bool_]
    reference_delay_s: FloatArray
    dispersion: ModalDispersion
    bulk_speed_m_s: float = float("nan")
    max_residual_wavenumber: float = float("nan")


class ContinuousModalPassiveSonarArraySimulator(ContinuousChannelFilterPassiveSonarArraySimulator):
    """Channel-filter simulator driven by a normal-mode sum, evaluated densely.

    Rendering is the same two-stage process as
    :class:`ContinuousChannelFilterPassiveSonarArraySimulator`: a true time-domain bulk
    delay read from an oversampled waveform, followed by short per-sensor filters that
    hold the rest of the channel, cross-faded between knots. What changes is where the
    channel comes from and how often it is evaluated.

    Modes are computed once
        The propagation model must be a :class:`KrakenModalPropagationModel`. Its modes
        are computed for the filter's frequency grid and the depth span the run needs,
        before anything is rendered. Each knot is then only a mode sum.

    Knots are dense and planned from the physics
        After the bulk delay ``r / c_ref`` is removed, mode ``m`` still turns in phase as
        ``(k_m - omega / c_ref) r``. Cross-fading two knots' filters interpolates linearly
        between them, which is accurate while that phase changes little between knots:
        a step of ``dphi`` errs by at most ``dphi**2 / 8`` of the mode's amplitude.
        Knots are spaced so it changes by at most ``max_phase_step_rad``, which gives
        each mode its own Doppler shift ``-k_m v_r`` and resolves modal fading as range
        changes. They are independent of the path's states, whose positions are
        interpolated.

    The bulk delay is chosen to need the fewest knots
        The modes that matter arrive between their slowest and fastest group speeds. A
        bulk delay ``r / c`` at a speed between those centres them in the filter, but the
        modes' phase speeds are faster still, so every mode's residual phase turns quickly
        and the knots must be close. Where the filter has room, ``c`` is moved towards the
        phase speeds, to the speed that minimises the fastest residual rotation while every
        arrival, the array's extent and the band taper's ring still fit within half the
        filter. Before rendering, the simulator checks that they fit at the centring speed
        and raises with a suggested ``filter_length`` if not. The knots are planned against
        whichever speed is used, so the interpolation error bound is the same, and the
        retarded geometry below uses that same speed, which is what keeps the Doppler exact
        to second order.

    Filters are streamed
        Each knot's filter is computed, used and discarded, so memory does not grow with
        the number of knots. :meth:`compute_channel_history` still returns the dense
        history for inspection or caching.

    With ``retarded_time`` the source position at each knot is taken at the time the
    sound left it, ``t - tau`` with ``tau`` the bulk delay, rather than when it arrives.
    The source sounds while the reception time lies within its path; before the path
    starts and after it ends the retarded position is held at the path's ends.
    """

    max_phase_step_rad: float = Property(
        default=np.pi / 16,
        doc="Largest change in any mode's residual phase between knots, in radians. Linear "
        "crossfading then errs by at most step^2 / 8 of the mode's amplitude: -46 dB at "
        "pi/16, -34 dB at pi/8",
    )
    knot_interval_s: float | None = Property(
        default=None, doc="Fixed knot interval; planned from max_phase_step_rad if None"
    )
    min_knot_interval_s: float = Property(default=0.02, doc="Shortest planned knot interval")
    max_knot_interval_s: float = Property(default=1.0, doc="Longest planned knot interval")
    geometry_step_s: float = Property(
        default=0.5, doc="Sampling of the geometry used to plan the knots, in seconds"
    )
    retarded_time: bool = Property(
        default=True, doc="Place the source where it was when the sound left it"
    )
    check_filter_length: bool = Property(
        default=True, doc="Raise if the modal arrivals do not fit in the filter"
    )
    optimise_bulk_speed: bool = Property(
        default=True,
        doc="Choose the bulk delay's speed to need the fewest knots while the arrivals still "
        "fit the filter; otherwise centre the arrivals on the group speeds",
    )
    delay_interpolation: str = Property(
        default="cubic",
        doc="Interpolation for the bulk delay; cubic by default, as the mode sum is exact "
        "enough for linear interpolation's loss of up to 0.17 dB to matter",
    )

    # ----------------------------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------------------------
    @property
    def _modal_model(self) -> KrakenModalPropagationModel:
        model = self.propagation_model
        if not isinstance(model, KrakenModalPropagationModel):
            raise TypeError(
                "ContinuousModalPassiveSonarArraySimulator needs a "
                f"KrakenModalPropagationModel, got {type(model).__name__}"
            )
        return model

    def _sensor_positions(self, times_s: FloatArray) -> FloatArray:
        """Sensor positions ``(T, 3, S)`` at times from ``start_time``."""
        times_s = np.asarray(times_s, dtype=np.float64).reshape(-1)
        history = getattr(self._array(), "platform_history", None)
        if history:
            # A towed array is only known at the times it was moved to; interpolate.
            cache = getattr(self, "_platform_cache", None)
            if cache is None or cache[0] is not history or cache[1] != len(history):
                ordered = sorted(history, key=lambda state: state.timestamp)
                stamps = np.array([self._elapsed_s(state.timestamp) for state in ordered])
                positions = np.stack(
                    [np.asarray(state.array.state_vector, dtype=np.float64) for state in ordered]
                )
                cache = (history, len(history), stamps, positions)
                self._platform_cache = cache
            stamps, positions = cache[2], cache[3]
            if stamps.size == 1:
                return np.repeat(positions, times_s.size, axis=0)
            upper = np.clip(np.searchsorted(stamps, times_s), 1, stamps.size - 1)
            lower = upper - 1
            weight = np.clip((times_s - stamps[lower]) / (stamps[upper] - stamps[lower]), 0.0, 1.0)
            return (
                positions[lower] * (1.0 - weight)[:, None, None]
                + positions[upper] * weight[:, None, None]
            )
        out = []
        for time_s in times_s:
            state = self._platform_state_at(self.start_time + timedelta(seconds=float(time_s)))
            out.append(np.asarray(state.array.state_vector, dtype=np.float64))
        return np.stack(out)

    def _path_track(self, path_index: int) -> tuple[FloatArray, FloatArray]:
        """Return a path's state times from ``start_time`` and positions ``(N, 3)``."""
        states = sorted(self.ground_truth_paths[path_index], key=lambda state: state.timestamp)
        if not states:
            raise ValueError(f"ground_truth_paths[{path_index}] has no states")
        times = np.array([self._elapsed_s(state.timestamp) for state in states])
        positions = np.stack([_get_source_position(state).reshape(-1)[:3] for state in states])
        return times, positions

    @staticmethod
    def _interpolate_track(
        track: tuple[FloatArray, FloatArray], times_s: FloatArray
    ) -> FloatArray:
        stamps, positions = track
        return np.stack([np.interp(times_s, stamps, positions[:, i]) for i in range(3)], axis=1)

    def _geometry(
        self, path_index: int, times_s: FloatArray, reference_speed_m_s: float | None
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        """Ranges ``(T, S)``, sensor depths ``(T, S)``, source depths and activity."""
        times_s = np.asarray(times_s, dtype=np.float64).reshape(-1)
        track = self._path_track(path_index)
        sensors = self._sensor_positions(times_s)
        source = self._interpolate_track(track, times_s)

        def horizontal(source_positions: FloatArray) -> FloatArray:
            offsets = sensors[:, :2, :] - source_positions[:, :2, np.newaxis]
            return np.hypot(offsets[:, 0], offsets[:, 1])

        ranges = horizontal(source)
        if self.retarded_time and reference_speed_m_s:
            # Two fixed-point steps of t_e = t - r(t_e) / c: errors shrink by v / c each.
            for _ in range(2):
                delay = np.median(ranges, axis=1) / reference_speed_m_s
                source = self._interpolate_track(track, times_s - delay)
                ranges = horizontal(source)
        tolerance = 1e-9
        active = (times_s >= track[0][0] - tolerance) & (times_s <= track[0][-1] + tolerance)
        model = self._modal_model
        ranges = np.maximum(ranges, float(model.min_range_m))
        return ranges, np.abs(sensors[:, 2, :]), np.abs(source[:, 2]), active

    # ----------------------------------------------------------------------------------
    # Planning
    # ----------------------------------------------------------------------------------
    def _coarse_times(self) -> FloatArray:
        step = max(float(self.geometry_step_s), 1e-3)
        duration = float(self.duration_s)
        return np.unique(np.concatenate((np.arange(0.0, duration, step), [duration])))

    def mode_set(self) -> ModeSet:
        """Modes for the filter's frequency grid over every depth the run needs.

        Computed on first use, covering all paths at once, and cached by the model.

        Returns
        -------
        ModeSet
            The modes.

        """
        times = self._coarse_times()
        depths = []
        max_range = 0.0
        speed = self._modal_model.kraken_environment().min_speed_m_s
        for index in range(len(self.ground_truth_paths or [])):
            ranges, sensor_depths, source_depths, _ = self._geometry(index, times, speed)
            # Retarded positions reach back along the path, so include its states too.
            path_depths = np.abs(self._path_track(index)[1][:, 2])
            depths += [sensor_depths.ravel(), source_depths, path_depths]
            max_range = max(max_range, float(ranges.max()))
        if not depths:
            raise ValueError("There are no ground-truth paths to simulate")
        return self._modal_model.modes_for(
            self.channel_frequencies_hz, np.concatenate(depths), rmax_km=max_range / 1e3
        )

    def plan_knots(self, path_index: int, modes: ModeSet | None = None) -> ModalKnots:
        """Plan one source's knots and evaluate its geometry at them.

        Parameters
        ----------
        path_index : int
            Index into ``ground_truth_paths``.
        modes : ModeSet, optional
            The run's modes; :meth:`mode_set` if omitted.

        Returns
        -------
        ModalKnots
            Knot times, geometry and bulk delays.

        Raises
        ------
        ValueError
            If ``check_filter_length`` is set and the modal arrivals do not fit in the
            filter.

        """
        model = self._modal_model
        modes = modes if modes is not None else self.mode_set()
        fs = self._sample_rate_hz
        coarse = self._coarse_times()
        speed = model.kraken_environment().min_speed_m_s
        ranges, _, source_depths, _ = self._geometry(path_index, coarse, speed)
        max_range = float(ranges.max())
        dispersion = model.dispersion(modes, float(np.median(source_depths)), max_range)
        # The group speeds' centre: the bulk speed that centres the arrivals in the filter.
        c_ref = dispersion.reference_speed_m_s
        ranges, _, _, _ = self._geometry(path_index, coarse, c_ref)
        reference = np.median(ranges, axis=1, keepdims=True)
        available = 0.5 * int(self.filter_length) / fs
        ring = 2.0 / max(float(self.band_taper_hz), 1e-9)

        def window(speed: float) -> float:
            """Half-width the arrivals need in the filter for a bulk delay at ``speed``."""
            late = ranges / dispersion.min_group_speed_m_s - reference / speed
            early = ranges / dispersion.max_group_speed_m_s - reference / speed
            return float(max(np.max(late), -np.min(early))) + ring

        # Each sensor's arrivals lie between r_s / v_max and r_s / v_min, which must fit about
        # the bulk delay r_ref / c. The band taper rings for about 2 / width.
        if self.check_filter_length:
            needed = window(c_ref)
            if needed > available:
                suggestion = int(2 ** np.ceil(np.log2(2.0 * needed * fs * 1.1)))
                raise ValueError(
                    f"Modal arrivals of source {path_index} spread over +/-{needed:.3f} s at "
                    f"{max_range / 1e3:.1f} km (group speeds {dispersion.min_group_speed_m_s:.1f}"
                    f"-{dispersion.max_group_speed_m_s:.1f} m/s), more than the filter's "
                    f"+/-{available:.3f} s; use filter_length={suggestion} or more"
                )

        significant = modes.significant_modes(
            float(np.median(source_depths)),
            max(max_range, 1.0),
            float(model.dispersion_threshold_db),
        )
        if not np.any(significant):
            significant = modes.mode_mask
        wavenumbers = modes.k.real[significant]
        omegas = np.broadcast_to(2.0 * np.pi * modes.frequencies_hz[:, None], modes.k.shape)[
            significant
        ]

        def residual(speed: float) -> float:
            return float(np.max(np.abs(wavenumbers - omegas / speed))) if wavenumbers.size else 0.0

        bulk_speed = c_ref
        if self.optimise_bulk_speed and wavenumbers.size:
            # The worst residual is a convex function of 1 / c, so a golden-section search
            # on slowness finds its minimum; the window constraint keeps it feasible.
            phase_speeds = omegas / wavenumbers
            lo, hi = 1.0 / float(np.max(phase_speeds)), 1.0 / c_ref
            if lo > hi:
                lo, hi = hi, lo
            ratio = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            for _ in range(60):
                x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
                if residual(1.0 / x1) <= residual(1.0 / x2):
                    b = x2
                else:
                    a = x1
            best = 1.0 / (0.5 * (a + b))
            if window(best) > 0.98 * available:
                # Walk back towards the centring speed until the arrivals fit.
                fractions = np.linspace(0.0, 1.0, 101)
                speeds = 1.0 / (1.0 / best + fractions * (1.0 / c_ref - 1.0 / best))
                fits = [speed for speed in speeds if window(speed) <= 0.98 * available]
                best = fits[0] if fits else c_ref
            if residual(best) < residual(c_ref):
                bulk_speed = best
        max_residual = residual(bulk_speed)

        if self.knot_interval_s is not None:
            interval = float(self.knot_interval_s)
        else:
            interval = self._planned_interval(coarse, ranges, reference, max_residual, bulk_speed)
        duration = float(self.duration_s)
        count = max(2, int(np.ceil(duration / interval)) + 1)
        knots = np.linspace(0.0, duration, count)
        # The retarded geometry must use the bulk delay's own speed. The residual filter
        # sees the bulk-delayed, Doppler-shifted waveform; its second-order Doppler terms
        # cancel only if the emission time is taken at that same speed. With the group
        # speed instead, a bulk speed nearer the phase speeds puts the source in the wrong
        # place by its speed times the difference in delay: -12 dB of error at 4 m/s.
        ranges, sensor_depths, source_depths, active = self._geometry(
            path_index, knots, bulk_speed
        )
        return ModalKnots(
            knot_times_s=knots,
            ranges_m=ranges,
            sensor_depths_m=sensor_depths,
            source_depths_m=source_depths,
            active=active,
            reference_delay_s=np.median(ranges, axis=1) / bulk_speed,
            dispersion=dispersion,
            bulk_speed_m_s=float(bulk_speed),
            max_residual_wavenumber=max_residual,
        )

    def _planned_interval(
        self,
        times: FloatArray,
        ranges: FloatArray,
        reference: FloatArray,
        max_residual_wavenumber: float,
        bulk_speed_m_s: float,
    ) -> float:
        """Knot interval keeping every mode's residual phase step within the limit."""
        if times.size < 2:
            return float(self.max_knot_interval_s)
        range_rate = np.abs(np.diff(ranges, axis=0)) / np.diff(times)[:, None]
        offset_rate = np.abs(np.diff(ranges - reference, axis=0)) / np.diff(times)[:, None]
        omega_max = 2.0 * np.pi * float(np.max(self.channel_frequencies_hz))
        phase_rate = max_residual_wavenumber * float(range_rate.max()) + (
            omega_max / bulk_speed_m_s
        ) * float(offset_rate.max())
        if phase_rate <= 0:
            return float(self.max_knot_interval_s)
        return float(
            np.clip(
                float(self.max_phase_step_rad) / phase_rate,
                float(self.min_knot_interval_s),
                float(self.max_knot_interval_s),
            )
        )

    # ----------------------------------------------------------------------------------
    # Channel and rendering
    # ----------------------------------------------------------------------------------
    def _knot_transfer(self, modes: ModeSet, knots: ModalKnots, knot: int) -> NDArray[Any]:
        """Transfer functions ``(S, F)`` at one knot; zero while the source is silent."""
        if not knots.active[knot]:
            return np.zeros(
                (knots.ranges_m.shape[1], modes.frequencies_hz.size), dtype=np.complex128
            )
        return modes.transfer(
            float(knots.source_depths_m[knot]),
            knots.sensor_depths_m[knot],
            knots.ranges_m[knot],
        )

    def compute_channel_history(self, path_index: int) -> ChannelHistory:
        """Evaluate one source's channel at every planned knot.

        This holds every knot's transfer function, ``K x S x F`` complex values, which the
        streaming renderer avoids. Use it to inspect or cache the channel.

        Parameters
        ----------
        path_index : int
            Index into ``ground_truth_paths``.

        Returns
        -------
        ChannelHistory
            Dense channel history; each sensor's delay is its range over the bulk speed,
            so the bulk delay it implies is the one the streaming renderer uses. With the
            bulk speed optimised that is not the physical arrival time.

        """
        modes = self.mode_set()
        knots = self.plan_knots(path_index, modes)
        transfer = np.stack(
            [self._knot_transfer(modes, knots, k) for k in range(knots.knot_times_s.size)]
        ).astype(np.complex64)
        return ChannelHistory(
            knot_times_s=knots.knot_times_s,
            transfer=transfer,
            delays_s=knots.ranges_m / knots.bulk_speed_m_s,
        )

    def _render_path(self, model: SourceSignal, path_index: int) -> ComplexArray:
        """Render one source, evaluating each knot's mode sum as it is needed."""
        if self.channel_histories is not None:
            return super()._render_path(model, path_index)
        modes = self.mode_set()
        knots = self.plan_knots(path_index, modes)
        length = int(self.filter_length)

        def filter_at(knot: int) -> ComplexArray:
            if not knots.active[knot]:
                return np.zeros((knots.ranges_m.shape[1], length), dtype=np.complex128)
            transfer = self._knot_transfer(modes, knots, knot)
            spectrum = self._filter_spectra(
                transfer[np.newaxis], knots.reference_delay_s[knot : knot + 1]
            )[0]
            return self._knot_filters(spectrum)

        return self._render_knots(
            model, path_index, knots.knot_times_s, knots.reference_delay_s, filter_at
        )

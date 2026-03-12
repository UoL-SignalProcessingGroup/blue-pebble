"""Discrete timestep anthropogenic signal models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property

from .base import Complex128Array, NarrowbandSignalBase, NarrowbandStatefulSignalBase

if TYPE_CHECKING:
    from stonesoup.types.state import State


class _BlendedSourceState(TypedDict):
    """Per-source continuity state for blended narrowband synthesis."""

    previous_signal: Complex128Array | None
    last_time: datetime | None
    cumulative_time: float


class _OverlapAddSourceState(TypedDict):
    """Per-source continuity state for overlap-add narrowband synthesis."""

    overlap_buffer: Complex128Array | None
    last_time: datetime | None
    cumulative_time: float


def _as_blended_source_state(state: dict[str, object]) -> _BlendedSourceState:
    """Validate and narrow a generic source-state mapping for blended synthesis."""
    previous_signal = state.get("previous_signal")
    if previous_signal is not None and not isinstance(previous_signal, np.ndarray):
        raise ValueError("Source state 'previous_signal' must be a NumPy array or None")
    return cast(_BlendedSourceState, state)


def _as_overlap_add_source_state(state: dict[str, object]) -> _OverlapAddSourceState:
    """Validate and narrow a generic source-state mapping for overlap-add synthesis."""
    overlap_buffer = state.get("overlap_buffer")
    if overlap_buffer is not None and not isinstance(overlap_buffer, np.ndarray):
        raise ValueError("Source state 'overlap_buffer' must be a NumPy array or None")
    return cast(_OverlapAddSourceState, state)


class NarrowbandTonalSignal(NarrowbandSignalBase):
    """Generate a narrowband tonal signal snapshot for each sensor."""

    def generate(
        self,
        source: State,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> Complex128Array:
        """Generate received narrowband tonal snapshots for all sensors.

        Parameters
        ----------
        source : State
            Source state with tonal metadata.
        sensor_delays_s : ArrayLike
            One-dimensional per-sensor delays in seconds.
        tloss_db : ArrayLike | float
            Scalar or one-dimensional transmission loss in dB.
        propagation_time_s : float
            Propagation time from source to array origin in seconds.

        Returns
        -------
        Complex128Array
            Complex sensor snapshot matrix with shape
            ``(num_sensors, num_samples)``.

        """
        time_array_s = self._build_time_array()
        amplitudes_upa, frequencies_hz, phases_rad = self._extract_tonal_metadata(source)
        received_amplitude_upa = self._scale_amplitudes_for_tloss(amplitudes_upa, tloss_db)

        return self._synthesise_sensor_signals(
            received_amplitude_upa,
            frequencies_hz,
            phases_rad,
            sensor_delays_s,
            propagation_time_s,
            time_array_s,
        )


class NarrowbandBlendedTonalSignal(NarrowbandStatefulSignalBase):
    """Generate narrowband tonal signals with overlap blending for continuity."""

    blend_fraction: float = Property(default=0.1, doc="Fraction of signal to blend for continuity")

    def generate(
        self,
        source: State,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> Complex128Array:
        """Generate narrowband snapshots with overlap blending continuity.

        Parameters
        ----------
        source : State
            Source state with tonal metadata and timestamp.
        sensor_delays_s : ArrayLike
            One-dimensional per-sensor delays in seconds.
        tloss_db : ArrayLike | float
            Scalar or one-dimensional transmission loss in dB.
        propagation_time_s : float
            Propagation time from source to array origin in seconds.

        Returns
        -------
        Complex128Array
            Complex sensor snapshot matrix with shape
            ``(num_sensors, num_samples)``.

        """
        num_sensors = len(sensor_delays_s)
        state = _as_blended_source_state(
            self._get_or_create_source_state(
                source,
                num_sensors,
                lambda: {
                    "previous_signal": None,
                    "last_time": None,
                    "cumulative_time": 0.0,
                },
            )
        )

        cumulative_time_s = self._update_cumulative_time_s(state, source)
        time_array_s = self._build_time_array(start_time_s=cumulative_time_s)

        amplitudes_upa, frequencies_hz, phases_rad = self._extract_tonal_metadata(source)
        received_amplitude_upa = self._scale_amplitudes_for_tloss(amplitudes_upa, tloss_db)

        sensor_signals = self._synthesise_sensor_signals(
            received_amplitude_upa,
            frequencies_hz,
            phases_rad,
            sensor_delays_s,
            propagation_time_s,
            time_array_s,
        )

        if state["previous_signal"] is not None:
            blend_fraction = float(self.blend_fraction)
            blend_samples = int(self.num_samples * blend_fraction)

            if blend_samples > 0:
                fade_out = np.cos(np.linspace(0, np.pi / 2, blend_samples)) ** 2
                fade_in = np.sin(np.linspace(0, np.pi / 2, blend_samples)) ** 2

                prev_tail = state["previous_signal"][:, -blend_samples:]
                current_head = sensor_signals[:, :blend_samples].copy()
                blended_section = (
                    prev_tail * fade_out[np.newaxis, :] + current_head * fade_in[np.newaxis, :]
                )
                sensor_signals[:, :blend_samples] = blended_section

        state["previous_signal"] = sensor_signals.copy()
        state["last_time"] = source.timestamp

        return sensor_signals


class NarrowbandOverlapAddTonalSignal(NarrowbandStatefulSignalBase):
    """Generate narrowband tonal signals using overlap-add reconstruction."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise overlap-add state and synthesis window cache."""
        super().__init__(*args, **kwargs)
        self._synthesis_window: NDArray[np.float64] | None = None

    def generate(
        self,
        source: State,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> Complex128Array:
        """Generate narrowband snapshots via overlap-add reconstruction.

        Parameters
        ----------
        source : State
            Source state with tonal metadata and timestamp.
        sensor_delays_s : ArrayLike
            One-dimensional per-sensor delays in seconds.
        tloss_db : ArrayLike | float
            Scalar or one-dimensional transmission loss in dB.
        propagation_time_s : float
            Propagation time from source to array origin in seconds.

        Returns
        -------
        Complex128Array
            Complex sensor snapshot matrix with shape
            ``(num_sensors, num_samples)``.

        """
        num_sensors = len(sensor_delays_s)
        state = _as_overlap_add_source_state(
            self._get_or_create_source_state(
                source,
                num_sensors,
                lambda: {
                    "overlap_buffer": None,
                    "last_time": None,
                    "cumulative_time": 0.0,
                },
            )
        )

        cumulative_time_s = self._update_cumulative_time_s(state, source)

        amplitudes_upa, frequencies_hz, phases_rad = self._extract_tonal_metadata(source)
        received_amplitude_upa = self._scale_amplitudes_for_tloss(amplitudes_upa, tloss_db)

        extended_samples = 2 * self.num_samples
        time_array_s = self._build_time_array(
            start_time_s=cumulative_time_s,
            num_samples=extended_samples,
        )

        sensor_signals_extended = self._synthesise_sensor_signals(
            received_amplitude_upa,
            frequencies_hz,
            phases_rad,
            sensor_delays_s,
            propagation_time_s,
            time_array_s,
        )

        synthesis_window = self._get_synthesis_window()
        sensor_signals_windowed = sensor_signals_extended * synthesis_window[np.newaxis, :]

        first_half = sensor_signals_windowed[:, : self.num_samples]
        second_half = sensor_signals_windowed[:, self.num_samples :]

        if state["overlap_buffer"] is not None:
            sensor_signals = first_half + state["overlap_buffer"]
        else:
            sensor_signals = first_half

        state["overlap_buffer"] = second_half.copy()
        state["last_time"] = source.timestamp

        return sensor_signals

    def reset(self) -> None:
        """Clear overlap-add state and synthesis window cache."""
        super().reset()
        self._synthesis_window = None

    def _get_synthesis_window(self) -> NDArray[np.float64]:
        """Generate or return the cached Hann synthesis window."""
        if self._synthesis_window is None:
            self._synthesis_window = np.hanning(2 * self.num_samples)
        return self._synthesis_window

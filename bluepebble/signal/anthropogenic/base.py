"""Shared abstractions for anthropogenic signal models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property

from ..base import ContinuousTimestepSignal, DiscreteTimestepSignal
from ..utils import compute_stft

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]


class NarrowbandSignalBase(DiscreteTimestepSignal, ABC):
    """Base class for narrowband anthropogenic signal models."""

    def _extract_tonal_metadata(
        self,
        source: Any,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Extract and validate tonal metadata from a source state."""
        metadata = getattr(source, "metadata", None)
        if metadata is None:
            msg = "Source state must define metadata for tonal synthesis"
            raise ValueError(msg)
        if not isinstance(metadata, Mapping):
            msg = "Source metadata must be mapping-like"
            raise ValueError(msg)

        required_keys = ("amplitudes_upa", "frequencies_hz", "phases_rad")
        missing_keys = [key for key in required_keys if key not in metadata]
        if missing_keys:
            missing = ", ".join(missing_keys)
            msg = f"Source metadata missing required keys: {missing}"
            raise ValueError(msg)

        amplitudes_upa = np.asarray(metadata["amplitudes_upa"], dtype=float)
        frequencies_hz = np.asarray(metadata["frequencies_hz"], dtype=float)
        phases_rad = np.asarray(metadata["phases_rad"], dtype=float)

        if amplitudes_upa.ndim != 1 or frequencies_hz.ndim != 1 or phases_rad.ndim != 1:
            msg = "Tonal metadata arrays must be one-dimensional"
            raise ValueError(msg)

        num_tonals = len(amplitudes_upa)
        if len(frequencies_hz) != num_tonals or len(phases_rad) != num_tonals:
            msg = (
                "Source tonal metadata arrays must have matching lengths: "
                f"len(amplitudes_upa)={len(amplitudes_upa)}, "
                f"len(frequencies_hz)={len(frequencies_hz)}, "
                f"len(phases_rad)={len(phases_rad)}"
            )
            raise ValueError(msg)

        return (
            cast(FloatArray, amplitudes_upa),
            cast(FloatArray, frequencies_hz),
            cast(FloatArray, phases_rad),
        )

    def _scale_amplitudes_for_tloss(
        self,
        amplitudes_upa: FloatArray,
        tloss_db: ArrayLike | float,
    ) -> FloatArray:
        """Apply transmission-loss attenuation to source amplitudes."""
        tloss_db_array = np.asarray(tloss_db, dtype=float)
        if tloss_db_array.ndim == 0:
            return cast(FloatArray, amplitudes_upa * 10 ** (-float(tloss_db_array) / 20.0))

        if tloss_db_array.ndim == 1:
            if len(tloss_db_array) != len(amplitudes_upa):
                raise ValueError(
                    f"Length of tloss_db array ({len(tloss_db_array)}) must match "
                    f"number of frequencies ({len(amplitudes_upa)})"
                )
            return cast(FloatArray, amplitudes_upa * 10 ** (-tloss_db_array / 20.0))

        msg = "tloss_db must be scalar-like or one-dimensional"
        raise ValueError(msg)

    def _build_time_array(
        self,
        start_time_s: float = 0.0,
        num_samples: int | None = None,
    ) -> FloatArray:
        """Construct a time array for tonal synthesis."""
        sample_count = self.num_samples if num_samples is None else int(num_samples)
        return cast(
            FloatArray,
            np.arange(sample_count, dtype=float) / self.sampling_rate_hz + float(start_time_s),
        )

    def _synthesise_sensor_signals(
        self,
        received_amplitude_upa: FloatArray,
        frequencies_hz: FloatArray,
        phases_rad: FloatArray,
        sensor_delays_s: ArrayLike,
        propagation_time_s: float,
        time_array_s: FloatArray,
    ) -> NDArray[np.complex128]:
        """Generate sensor-array tonal signals via vectorised broadcasting."""
        sensor_delays_array = np.asarray(sensor_delays_s, dtype=float)
        if sensor_delays_array.ndim != 1:
            msg = "sensor_delays_s must be one-dimensional"
            raise ValueError(msg)

        time_reshaped = time_array_s[np.newaxis, np.newaxis, :]
        delays_reshaped = sensor_delays_array[:, np.newaxis, np.newaxis]
        freq_reshaped = frequencies_hz[np.newaxis, :, np.newaxis]
        phase_reshaped = phases_rad[np.newaxis, :, np.newaxis]

        total_phase = (
            2 * np.pi * freq_reshaped * (time_reshaped - propagation_time_s - delays_reshaped)
            + phase_reshaped
        )

        amp_reshaped = received_amplitude_upa[np.newaxis, :, np.newaxis]
        all_tonal_components = amp_reshaped * np.exp(1j * total_phase)

        return cast(NDArray[np.complex128], np.sum(all_tonal_components, axis=1))


class NarrowbandStatefulSignalBase(NarrowbandSignalBase, ABC):
    """Base class for narrowband signal models requiring per-source state."""

    def __init__(self, *args, **kwargs):
        """Initialise stateful narrowband signal model storage."""
        super().__init__(*args, **kwargs)
        self._source_states: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}

    def _get_source_key(
        self,
        source: Any,
        num_sensors: int,
    ) -> tuple[tuple[float, ...], int]:
        """Generate a hashable state key for a source and sensor configuration."""
        frequencies_hz = np.asarray(source.metadata["frequencies_hz"], dtype=float)
        return (tuple(frequencies_hz.tolist()), int(num_sensors))

    def _get_or_create_source_state(
        self,
        source: Any,
        num_sensors: int,
        default_factory: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Fetch per-source state, creating it when first seen."""
        source_key = self._get_source_key(source, num_sensors)
        if source_key not in self._source_states:
            self._source_states[source_key] = default_factory()
        return self._source_states[source_key]

    def _update_cumulative_time_s(
        self,
        state: MutableMapping[str, Any],
        source: Any,
    ) -> float:
        """Update cumulative synthesis time for phase continuity."""
        source_time = getattr(source, "timestamp", None)
        if source_time is None:
            msg = "Source state must define a timestamp for stateful synthesis"
            raise ValueError(msg)

        if "cumulative_time" not in state:
            state["cumulative_time"] = 0.0

        last_time = state.get("last_time")
        if last_time is not None:
            time_delta = (source_time - last_time).total_seconds()
            state["cumulative_time"] += time_delta

        return state["cumulative_time"]

    def reset(self):
        """Clear stored per-source state."""
        self._source_states.clear()


class BroadbandStftSignalBase(ContinuousTimestepSignal, ABC):
    """Base class for broadband STFT-first anthropogenic signal models."""

    frame_len = Property(int, default=1024, doc="STFT frame length in samples")
    hop_factor = Property(int, default=4, doc="Hop factor (hop = frame_len // hop_factor)")
    window_type = Property(str, default="hann", doc="Window type for STFT")

    def __init__(self, *args, **kwargs):
        """Initialise shared broadband STFT caches."""
        super().__init__(*args, **kwargs)
        self._stft_cache: NDArray[np.complex64] | None = None
        self._frequencies: FloatArray | None = None
        self._hop: int | None = None
        self._window: FloatArray | None = None
        self._source_signal: ComplexArray | None = None

    @abstractmethod
    def _generate_source_signal(self, source: Any) -> ComplexArray:
        """Generate a full-length source signal for STFT processing."""

    def compute_stft(
        self, source: Any
    ) -> tuple[NDArray[np.complex64], FloatArray, int, FloatArray]:
        """Compute and cache the source STFT and associated parameters."""
        if self._stft_cache is not None:
            return (
                self._stft_cache,
                cast(FloatArray, self._frequencies),
                cast(int, self._hop),
                cast(FloatArray, self._window),
            )

        self._source_signal = self._generate_source_signal(source)

        stft, freq_normalized, hop, window = compute_stft(
            self._source_signal, self.frame_len, self.hop_factor, self.window_type
        )
        frequencies = cast(FloatArray, freq_normalized * self.sampling_rate_hz)

        self._stft_cache = stft
        self._frequencies = frequencies
        self._hop = hop
        self._window = window

        return stft, frequencies, hop, window

    def get_stft(self) -> tuple[NDArray[np.complex64], FloatArray, int, FloatArray]:
        """Return cached STFT data."""
        if self._stft_cache is None:
            msg = "STFT not computed yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return (
            self._stft_cache,
            cast(FloatArray, self._frequencies),
            cast(int, self._hop),
            cast(FloatArray, self._window),
        )

    def get_source_signal(self) -> ComplexArray:
        """Return the cached source signal."""
        if self._source_signal is None:
            msg = "Source signal not generated yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._source_signal

    def generate(
        self,
        source: Any,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> NDArray[np.complex128]:
        """Broadband models operate in STFT space and do not support per-step generation."""
        _ = source, sensor_delays_s, tloss_db, propagation_time_s
        msg = (
            f"{type(self).__name__} does not support per-timestep generation. "
            "Use compute_stft() and process in frequency domain via "
            "ContinuousPassiveSonarArraySimulator."
        )
        raise NotImplementedError(msg)

    def reset(self):
        """Clear cached STFT and source-signal state."""
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None

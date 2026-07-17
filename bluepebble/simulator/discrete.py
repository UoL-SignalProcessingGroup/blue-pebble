"""Discrete acoustic sensor simulators module."""

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property

from ..models.propagation import SpectrumPropagationModel
from ..signal.biological import BiologicalSignal
from .base import PassiveSonarArraySimulatorBase, SensorBatch

if TYPE_CHECKING:
    from stonesoup.types.state import State

Complex64Array: TypeAlias = NDArray[np.complex64]
Complex128Array: TypeAlias = NDArray[np.complex128]


class DiscretePassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Discrete broadband simulator with per-timestamp spectrum rendering.

    This simulator synthesises one independent snapshot per platform timestamp.
    It is intended for broadband scenarios where each timestep can be processed
    as a standalone frame, without enforcing waveform continuity across adjacent
    timestamps.

    Processing stages
    -----------------
    1. Resolve one signal model per target (or broadcast a single shared model).
    2. Build one full source waveform per target via ``get_source_waveform``.
    3. Partition source waveforms into timestamp-aligned chunks using platform
       time spacing and sampling rate.
    4. For each timestamp, evaluate ``H(f)`` from ``propagate_spectrum`` at the
       chunk FFT bins for each active target.
    5. Apply ``H(f)`` in the frequency domain and IFFT per sensor, summing
       contributions from all targets.
    6. Add optional ambient noise, run optional beamforming, and emit one
       ``PassiveSonarSensorData`` payload.

    Notes
    -----
    - This method does not interpolate transfer functions across timesteps.
    - No overlap-add or continuity smoothing is applied between emitted chunks.
    - Timestamp boundaries that produce zero-length chunks are clamped to a
      minimum one-sample snapshot to keep FFT processing valid.
    - **No Doppler is modelled**: chunks are rendered independently with no cross-timestep
      continuity. Use
      :class:`~bluepebble.simulator.continuous.ContinuousFractionalDelayPassiveSonarArraySimulator`
      when Doppler matters.

    Tradeoffs
    ---------
    - Lower complexity and simpler reasoning than continuous STFT/WOLA methods.
    - Best suited to analyses where per-timestep independence is acceptable.

    """

    signal_models: list[BiologicalSignal] = Property(
        doc="List of broadband signal models (one per target, or single-element list for all)",
    )

    def _resolve_signal_models(self, num_targets: int) -> list[BiologicalSignal]:
        """Resolve one signal model per target.

        Parameters
        ----------
        num_targets : int
            Number of targets represented in the current scenario.

        Returns
        -------
        list of Signal
            Resolved signal-model list where each target has one model.

        """
        return self._resolve_models(self.signal_models, num_targets, "signal models")

    @staticmethod
    def _get_target_first_state(target_path: "Iterable[State]") -> "State | None":
        """Return the first state from a target path.

        Parameters
        ----------
        target_path : Iterable of State
            Target state sequence.

        Returns
        -------
        State or None
            First state if available, else ``None`` for empty paths.

        """
        try:
            return next(iter(target_path))
        except StopIteration:
            return None

    @staticmethod
    def _get_broadband_source_signal(
        signal_model: BiologicalSignal,
        first_state: "State | None",
    ) -> Complex128Array:
        """Get a source waveform from supported signal-model interfaces.

        Parameters
        ----------
        signal_model : Signal
            Signal model whose ``get_source_waveform`` returns the full-duration
            source waveform.
        first_state : State or None
            First target state, used to initialise lazy source generation.

        Returns
        -------
        numpy.ndarray
            Complex source waveform as ``complex128``.

        Raises
        ------
        ValueError
            If no target state is available to initialise the source waveform.

        """
        if first_state is None:
            msg = (
                "Cannot initialise broadband source signal for an empty target path. "
                "Provide a target state or pre-compute the source signal."
            )
            raise ValueError(msg)
        return np.asarray(signal_model.get_source_waveform(first_state), dtype=np.complex128)

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield one independent broadband snapshot per platform timestamp.

        Yields
        ------
        tuple of (datetime, set of PassiveSonarSensorData)
            Timestamp and simulated sensor-data set for that timestamp.

        Raises
        ------
        AttributeError
            If the configured propagation model does not implement
            ``propagate_spectrum``.
        ValueError
            If signal model configuration is invalid.

        """
        if not isinstance(self.propagation_model, SpectrumPropagationModel):
            msg = (
                f"{type(self.propagation_model).__name__} does not implement "
                "propagate_spectrum; use a SpectrumPropagationModel subclass"
            )
            raise AttributeError(msg)
        spectrum_propagation_model = cast(SpectrumPropagationModel, self.propagation_model)

        all_timestamps = self._sorted_timestamps()

        if not all_timestamps:
            msg = (
                "platform has no movement states; call platform.move() before running the "
                "simulator"
            )
            raise ValueError(msg)

        ground_truth_paths = self.ground_truth_paths or []
        signal_models_list = self._resolve_signal_models(len(ground_truth_paths))

        if len(signal_models_list) == 0:
            msg = "signal models must contain at least one model"
            raise ValueError(msg)

        num_sensors = int(self.platform.num_sensors)
        sampling_rate_hz = float(signal_models_list[0].sampling_rate_hz)
        total_samples = int(signal_models_list[0].num_samples)

        if total_samples <= 0:
            msg = "signal model num_samples must be greater than zero"
            raise ValueError(msg)

        t0 = all_timestamps[0]
        step_times_s = np.array(
            [(ts - t0).total_seconds() for ts in all_timestamps],
            dtype=np.float64,
        )
        step_sample_idx = np.rint(step_times_s * sampling_rate_hz).astype(np.int64)
        step_sample_idx = np.clip(step_sample_idx, 0, total_samples)
        step_sample_idx = np.maximum.accumulate(step_sample_idx)

        source_signal_by_target: list[Complex64Array] = []
        for target_idx, target_path in enumerate(ground_truth_paths):
            target_signal_model = signal_models_list[target_idx]

            if int(target_signal_model.num_samples) != total_samples:
                msg = (
                    "All signal models must share the same num_samples for discrete "
                    "broadband simulation."
                )
                raise ValueError(msg)

            if float(target_signal_model.sampling_rate_hz) != sampling_rate_hz:
                msg = (
                    "All signal models must share the same sampling_rate_hz for "
                    "discrete broadband simulation."
                )
                raise ValueError(msg)

            first_state = self._get_target_first_state(target_path)
            source_signal = self._get_broadband_source_signal(target_signal_model, first_state)

            if len(source_signal) < total_samples:
                pad_len = total_samples - len(source_signal)
                source_signal = np.concatenate(
                    [source_signal, np.zeros(pad_len, dtype=np.complex64)]
                )
            elif len(source_signal) > total_samples:
                source_signal = source_signal[:total_samples]

            source_signal_by_target.append(np.asarray(source_signal, dtype=np.complex64))

        n_steps = len(all_timestamps)
        for step_idx, timestamp in enumerate(all_timestamps):
            platform_state = self.platform.get_platform_state_at(timestamp)

            start_sample = int(step_sample_idx[step_idx])
            if step_idx < n_steps - 1:
                end_sample = int(step_sample_idx[step_idx + 1])
            else:
                end_sample = total_samples

            # Ensure a non-empty chunk for FFT processing.
            if end_sample <= start_sample:
                if start_sample >= total_samples:
                    start_sample = max(0, total_samples - 1)
                    end_sample = total_samples
                else:
                    end_sample = min(total_samples, start_sample + 1)

            num_samples_snapshot = end_sample - start_sample
            frequencies_hz = np.fft.fftfreq(num_samples_snapshot, d=1.0 / sampling_rate_hz)
            sensor_signals = np.zeros((num_sensors, num_samples_snapshot), dtype=np.complex64)

            for target_idx, target_path in enumerate(ground_truth_paths):
                target_state = self._target_state_at(target_path, timestamp)
                if target_state is None:
                    continue

                source_chunk = source_signal_by_target[target_idx][start_sample:end_sample]
                source_fft = np.fft.fft(source_chunk)

                H_sensors, _ = spectrum_propagation_model.propagate_spectrum(
                    platform_state,
                    target_state,
                    frequencies_hz,
                )
                target_fft = np.asarray(H_sensors, dtype=np.complex64) * source_fft[np.newaxis, :]
                sensor_signals += np.fft.ifft(target_fft, axis=1).astype(np.complex64)

            noise = self._generate_noise(
                num_sensors=num_sensors,
                num_samples=num_samples_snapshot,
            )
            if noise is not None:
                sensor_signals += noise

            beamformed_data = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
            )
            yield timestamp, {sensor_data}

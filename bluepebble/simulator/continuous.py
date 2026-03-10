"""Continuous acoustic sensor simulators module."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

from .base import PassiveSonarArraySimulatorBase


class ContinuousPassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """STFT-interpolated broadband passive-sonar simulator.

    This simulator implements an STFT-domain propagation workflow intended for continuous
    broadband scenarios with moving source/receiver geometry. Unlike snapshot simulators that
    synthesize each timestamp independently, this class renders one continuous receive sequence
    per sensor and then slices it into timestamped outputs.

    Processing stages
    -----------------
    1. Build a source STFT per target.
    2. Sample ``H(f)`` from ``propagate_spectrum`` at each simulation timestamp.
    3. Linearly interpolate ``H(f)`` across frame centers.
    4. Apply channel response to each target STFT and sum targets in the frequency domain.
    5. Reconstruct per-sensor time signals and slice by timestamp boundaries.

    Notes
    -----
    - Interpolation is delay-aware: dominant phase is de-rotated before interpolation and
        re-applied afterwards.
    - For stronger phase-stability in rapidly varying channels, prefer
      ``BroadbandWOLAPassiveSonarArraySimulator``.
    """

    signal_models = Property(
        list,
        doc="List of broadband signal models (one per target, or single-element list for all)",
    )
    fade_in_ms = Property(float, default=1000.0, doc="Fade-in duration at arrival (ms)")

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate continuous broadband sensor data using STFT processing."""
        all_timestamps = self._sorted_timestamps()
        if len(all_timestamps) < 2:
            msg = "Need at least 2 timesteps for broadband processing"
            raise ValueError(msg)

        ground_truth_paths = self.ground_truth_paths or []
        if len(ground_truth_paths) == 0:
            msg = "ContinuousPassiveSonarArraySimulator requires at least one target"
            raise ValueError(msg)

        signal_models_list = self._resolve_models(
            self.signal_models,
            len(ground_truth_paths),
            "signal models",
        )

        from bluepebble.signal.utils import apply_fade_in, inverse_stft

        first_target_path = ground_truth_paths[0]
        first_state = next(iter(first_target_path))
        source_stft, frequencies, hop, window = signal_models_list[0].compute_stft(first_state)
        num_frames, num_freq_bins = source_stft.shape
        source_signal = signal_models_list[0].get_source_signal()
        n_steps = len(all_timestamps)
        sampling_rate_hz = float(signal_models_list[0].sampling_rate_hz)

        if n_steps > 1:
            step_duration_s = (all_timestamps[1] - all_timestamps[0]).total_seconds()
        else:
            step_duration_s = len(source_signal) / sampling_rate_hz

        num_sensors = self.platform.num_sensors
        targets_data = []
        for target_idx, target_path in enumerate(ground_truth_paths):
            target_first_state = next(iter(target_path))
            target_signal_model = signal_models_list[target_idx]
            target_source_stft, _, _, _ = target_signal_model.compute_stft(target_first_state)

            H_list_all = []
            tdelay_sensor_list = []
            for timestamp in all_timestamps:
                platform_state = self.platform.get_platform_state_at(timestamp)
                target_state = self._target_state_at(target_path, timestamp)

                if target_state is None:
                    H_list_all.append(
                        np.zeros((num_sensors, len(frequencies)), dtype=np.complex64)
                    )
                    tdelay_sensor_list.append(np.zeros(num_sensors, dtype=np.float64))
                    continue

                H_sensors, prop_time_s = self.propagation_model.propagate_spectrum(
                    platform_state,
                    target_state,
                    frequencies,
                )
                H_list_all.append(np.asarray(H_sensors, dtype=np.complex64))

                if hasattr(self.propagation_model, "compute_sensor_delays"):
                    sensor_delays_s = self.propagation_model.compute_sensor_delays(
                        platform_state,
                        target_state,
                    )
                else:
                    sensor_delays_s = np.zeros(num_sensors, dtype=np.float64)

                tdelay_sensors = np.asarray(prop_time_s + sensor_delays_s, dtype=np.float64)
                tdelay_sensor_list.append(tdelay_sensors)

            targets_data.append(
                {
                    "H_list": H_list_all,
                    "tdelay_sensor_list": tdelay_sensor_list,
                    "source_stft": target_source_stft,
                }
            )

        receiver_signals = []
        # Reverse sensor order to preserve historical beamformer channel ordering.
        for sensor_idx in reversed(range(num_sensors)):
            stft_out_total = np.zeros((num_frames, num_freq_bins), dtype=np.complex64)

            for target_data in targets_data:
                H_sensor_history = np.asarray(
                    [H_list[sensor_idx, :] for H_list in target_data["H_list"]],
                    dtype=np.complex64,
                )
                tdelay_history = np.asarray(
                    [delays[sensor_idx] for delays in target_data["tdelay_sensor_list"]],
                    dtype=np.float64,
                )
                target_source_stft = np.asarray(target_data["source_stft"], dtype=np.complex64)

                frame_times_s = (
                    np.arange(num_frames, dtype=np.float64) * hop + hop // 2
                ) / sampling_rate_hz
                step_idx_float = frame_times_s / max(step_duration_s, 1e-12)
                step_idx = np.floor(step_idx_float).astype(np.int32)
                step_idx = np.clip(step_idx, 0, n_steps - 2)
                alpha = np.clip(step_idx_float - step_idx, 0.0, 1.0)

                H_current = H_sensor_history[step_idx, :]
                H_next = H_sensor_history[step_idx + 1, :]
                H_interp = H_current * (1.0 - alpha)[:, np.newaxis] + H_next * alpha[:, np.newaxis]

                t_current = tdelay_history[step_idx]
                t_next = tdelay_history[step_idx + 1]
                t_interp = t_current * (1.0 - alpha) + t_next * alpha
                phase = np.exp(
                    -2j
                    * np.pi
                    * t_interp[:, np.newaxis]
                    * np.asarray(frequencies, dtype=np.float64)[np.newaxis, :]
                )

                stft_out_total += target_source_stft * H_interp * phase

            signal_reconstructed = inverse_stft(
                stft_out_total,
                signal_models_list[0].frame_len,
                hop,
                window,
            )
            if self.fade_in_ms > 0:
                fade_samples = int(self.fade_in_ms * sampling_rate_hz / 1000.0)
                signal_with_arrival = apply_fade_in(signal_reconstructed, fade_samples)
            else:
                signal_with_arrival = signal_reconstructed
            receiver_signals.append(np.asarray(signal_with_arrival, dtype=np.complex64))

        max_len = max(len(signal) for signal in receiver_signals)
        for idx, signal in enumerate(receiver_signals):
            if len(signal) < max_len:
                pad_len = max_len - len(signal)
                receiver_signals[idx] = np.concatenate(
                    [signal, np.zeros(pad_len, dtype=np.complex64)],
                )

        receiver_signals_array = np.asarray(receiver_signals, dtype=np.complex64)
        actual_signal_len = receiver_signals_array.shape[1]
        samples_per_step = max(actual_signal_len // n_steps, 1)

        for step_idx, timestamp in enumerate(all_timestamps):
            start_sample = step_idx * samples_per_step
            if step_idx == n_steps - 1:
                end_sample = actual_signal_len
            else:
                end_sample = start_sample + samples_per_step
            sensor_signals = receiver_signals_array[:, start_sample:end_sample]

            noise = self._generate_noise(
                num_sensors=num_sensors,
                num_samples=sensor_signals.shape[1],
                sampling_rate_hz=sampling_rate_hz,
            )
            if noise is not None:
                sensor_signals = sensor_signals + noise

            sensor_signals_for_beamformer = np.real(sensor_signals).astype(np.complex128)
            beamformed_data = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals_for_beamformer,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
            )

            yield timestamp, {sensor_data}


            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
            )
            yield timestamp, {sensor_data}

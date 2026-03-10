"""Continuous acoustic sensor simulatiors module."""

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
    - Sensor output ordering is reversed during reconstruction to preserve expected
      beamformer channel ordering.
    - For stronger phase-stability in rapidly varying channels, prefer
      ``BroadbandWOLAPassiveSonarArraySimulator``.
    """

    signal_models = Property(
        list,
        doc="List of broadband signal models (one per target, or single-element list for all)",
    )
    fade_in_ms = Property(float, default=1000.0, doc="Fade-in duration at arrival (ms)")

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate continuous broadband sensor data using STFT processing.

        This generator implements the BroadbandArrayProcessor methodology:
            1. Compute source STFT once
            2. For each timestep, get transfer functions H(f)
            3. Interpolate H(f) between timesteps
            4. Apply to STFT frames and reconstruct per sensor

        Yields
        ------
        tuple
            (timestamp, set of PassiveSonarSensorData) for each timestep.

        """
        # Import STFT utilities
        from bluepebble.signal.utils import apply_fade_in, inverse_stft

        # Get all timestamps
        all_timestamps = self._sorted_timestamps()

        if len(all_timestamps) < 2:
            msg = "Need at least 2 timesteps for broadband processing"
            raise ValueError(msg)

        # Get all target paths
        ground_truth_paths = self.ground_truth_paths or []
        if len(ground_truth_paths) == 0:
            msg = "BroadbandPassiveSonarArraySimulator requires at least one target"
            raise ValueError(msg)

        # Normalize signal_models to list (support single or per-target)
        signal_models_list = self._resolve_models(
            self.signal_models,
            len(ground_truth_paths),
            "signal models",
        )

        # Get first target state from first path to initialize STFT parameters
        first_target_path = ground_truth_paths[0]
        first_state = next(iter(first_target_path))

        # Compute STFT of first source signal to get parameters
        # (all signal models should have same STFT parameters)
        source_stft, frequencies, hop, window = signal_models_list[0].compute_stft(first_state)
        num_frames = source_stft.shape[0]
        num_freq_bins = source_stft.shape[1]

        # Get source time-domain signal for reference
        source_signal = signal_models_list[0].get_source_signal()

        # Calculate timestep parameters
        total_duration_s = len(source_signal) / signal_models_list[0].sampling_rate_hz
        n_steps = len(all_timestamps)
        if n_steps > 1:
            step_duration_s = (all_timestamps[1] - all_timestamps[0]).total_seconds()
        else:
            step_duration_s = total_duration_s

        num_sensors = self.platform.num_sensors

        # Storage for transfer functions at each timestep for each target
        # Structure: list of dicts, one dict per target containing:
        #   - 'H_list': list of (num_sensors, num_frequencies) per timestep
        #   - 'tdelay_sensor_list': list of per-sensor propagation delays per timestep
        #   - 'source_stft': STFT of this target's source signal
        targets_data = []

        # Process each target
        for target_idx, target_path in enumerate(ground_truth_paths):
            # Get first state to generate STFT for this target
            target_first_state = next(iter(target_path))

            # Get signal model for this specific target
            target_signal_model = signal_models_list[target_idx]

            # Compute STFT for this target's unique source signal
            target_source_stft, _, _, _ = target_signal_model.compute_stft(target_first_state)

            H_list_all = []  # List of (num_sensors, num_frequencies) per timestep
            tdelay_sensor_list = []  # List of per-sensor delays per timestep

            # Run propagation simulation for each timestep to get H(f)
            for timestamp in all_timestamps:
                # Get platform state
                platform_state = self.platform.get_platform_state_at(timestamp)

                # Get target state at this timestamp
                target_state = self._target_state_at(target_path, timestamp)

                if target_state is None:
                    # Keep list lengths aligned with timestamps.
                    # Absent target contributes zero transfer for this timestep.
                    H_list_all.append(np.zeros((num_sensors, len(frequencies)), dtype=np.complex64))
                    tdelay_sensor_list.append(np.zeros(num_sensors, dtype=np.float64))
                    continue

                # Run spectrum propagation to get H(f) for all sensors
                # NOTE: H_sensors already contains full phase information from rtrs,
                # including propagation delay, multipath interference, and caustics.
                # No additional phase shift is needed.
                H_sensors, prop_time_s = self.propagation_model.propagate_spectrum(
                    platform_state, target_state, frequencies
                )

                # H_sensors shape: (num_sensors, num_frequencies)
                H_list_all.append(H_sensors)

                # Build per-sensor absolute delay history for robust de-rotation.
                # compute_sensor_delays returns delays relative to array reference.
                sensor_delays_s = self.propagation_model.compute_sensor_delays(
                    platform_state,
                    target_state,
                )
                tdelay_sensors = np.asarray(prop_time_s + sensor_delays_s, dtype=np.float64)
                tdelay_sensor_list.append(tdelay_sensors)

            # Store this target's data
            targets_data.append(
                {
                    "H_list": H_list_all,
                    "tdelay_sensor_list": tdelay_sensor_list,
                    "source_stft": target_source_stft,
                }
            )

        # Now reconstruct signals for each sensor using overlap-add
        # For multiple targets, sum contributions in the frequency domain
        receiver_signals = []

        for sensor_idx in range(num_sensors):
            # Accumulate STFT output from all targets
            STFT_out_total = np.zeros((num_frames, num_freq_bins), dtype=np.complex64)

            # Process each target
            for target_data in targets_data:
                H_list_all = target_data["H_list"]
                tdelay_sensor_list = target_data["tdelay_sensor_list"]
                target_source_stft = target_data["source_stft"]

                if len(H_list_all) != n_steps:
                    msg = (
                        f"Transfer-function history length ({len(H_list_all)}) does not match "
                        f"number of timesteps ({n_steps})."
                    )
                    raise RuntimeError(msg)

                if len(tdelay_sensor_list) != n_steps:
                    msg = (
                        f"Propagation-delay history length ({len(tdelay_sensor_list)}) does not "
                        f"match number of timesteps ({n_steps})."
                    )
                    raise RuntimeError(msg)

                # Extract transfer function history for this sensor and target
                H_sensor_history = np.asarray(
                    [H_list[sensor_idx, :] for H_list in H_list_all],
                    dtype=np.complex64,
                )

                # Prepare interpolation indices once per target
                frame_times_s = (
                    (np.arange(num_frames, dtype=np.float64) * hop + hop // 2)
                    / signal_models_list[0].sampling_rate_hz
                )
                step_idx_float = frame_times_s / max(step_duration_s, 1e-12)
                step_idx = np.floor(step_idx_float).astype(np.int32)
                step_idx = np.clip(step_idx, 0, n_steps - 2)
                alpha = np.clip(step_idx_float - step_idx, 0.0, 1.0)

                # Interpolate H(f) across time for each STFT frame
                STFT_out_target = np.zeros((num_frames, num_freq_bins), dtype=np.complex64)

                # De-rotate with known propagation delay at each timestep to remove
                # the dominant high-rate phase term before interpolation.
                # This prevents phase-branch aliasing when target/platform motion is
                # fast relative to coarse simulator timesteps.
                tdelay_history = np.asarray(
                    [delays[sensor_idx] for delays in tdelay_sensor_list],
                    dtype=np.float64,
                )
                freq_axis = np.asarray(frequencies, dtype=np.float64)

                phase_derotate = np.exp(
                    2j
                    * np.pi
                    * tdelay_history[:, np.newaxis]
                    * freq_axis[np.newaxis, :]
                )
                H_residual = H_sensor_history * phase_derotate

                # Interpolate residual transfer functions in magnitude/phase domain
                # to avoid chord artifacts from direct complex interpolation.
                H_mag = np.abs(H_residual)
                H_phase = np.unwrap(np.angle(H_residual), axis=0)

                H_mag_interp = (
                    H_mag[step_idx, :] * (1.0 - alpha)[:, np.newaxis]
                    + H_mag[step_idx + 1, :] * alpha[:, np.newaxis]
                )
                H_phase_interp = (
                    H_phase[step_idx, :] * (1.0 - alpha)[:, np.newaxis]
                    + H_phase[step_idx + 1, :] * alpha[:, np.newaxis]
                )

                # Re-apply interpolated delay phase term.
                tdelay_interp = (
                    tdelay_history[step_idx] * (1.0 - alpha)
                    + tdelay_history[step_idx + 1] * alpha
                )
                phase_rerotate = np.exp(
                    -2j * np.pi * tdelay_interp[:, np.newaxis] * freq_axis[np.newaxis, :]
                )
                H_interp = H_mag_interp * np.exp(1j * H_phase_interp) * phase_rerotate

                # Apply transfer function to this target's source STFT
                STFT_out_target[:, :] = target_source_stft * H_interp

                # Add this target's contribution to total
                STFT_out_total += STFT_out_target

            # Reconstruct time-domain signal using inverse STFT (sum of all targets)
            signal_reconstructed = inverse_stft(
                STFT_out_total, signal_models_list[0].frame_len, hop, window
            )

            # The delay has been handled in the frequency domain via phase shift.
            # We only need to apply a fade-in if specified (to smooth the arrival).
            if self.fade_in_ms > 0:
                fade_samples = int(
                    self.fade_in_ms * signal_models_list[0].sampling_rate_hz / 1000.0
                )
                signal_with_arrival = apply_fade_in(signal_reconstructed, fade_samples)
            else:
                signal_with_arrival = signal_reconstructed

            receiver_signals.append(signal_with_arrival)

        # Ensure all signals have the same length
        max_len = max(len(sig) for sig in receiver_signals)
        for i in range(num_sensors):
            if len(receiver_signals[i]) < max_len:
                pad_len = max_len - len(receiver_signals[i])
                receiver_signals[i] = np.concatenate(
                    [receiver_signals[i], np.zeros(pad_len, dtype=np.complex64)]
                )

        receiver_signals_array = np.array(receiver_signals)  # Shape: (num_sensors, total_samples)

        # Now yield sensor data for each timestep by slicing the continuous signals
        # Account for trimmed signal length from inverse_stft
        actual_signal_len = receiver_signals_array.shape[1]
        samples_per_step = actual_signal_len // n_steps

        for step_idx, timestamp in enumerate(all_timestamps):
            # Extract signal slice for this timestep
            start_sample = step_idx * samples_per_step
            end_sample = start_sample + samples_per_step

            # Ensure we don't exceed array bounds on last timestep
            if step_idx == n_steps - 1:
                end_sample = actual_signal_len

            sensor_signals = receiver_signals_array[:, start_sample:end_sample]

            # Add noise if provided
            actual_samples = sensor_signals.shape[1]
            noise = self._generate_noise(
                num_sensors=num_sensors,
                num_samples=actual_samples,
                sampling_rate_hz=signal_models_list[0].sampling_rate_hz,
            )
            if noise is not None:
                sensor_signals += noise

            # Apply beamforming if provided
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


                platform_state = self.platform.get_platform_state_at(timestamp)
                steering_delays_s = self.steering_calculator.calculate(platform_state)

                # Convert complex signals to real for beamforming
                # Broadband signals are in baseband (complex) representation
                sensor_signals_real = np.real(sensor_signals).astype(np.complex128)
                beamformed_data = self.beamformer.beamform(sensor_signals_real, steering_delays_s)

            # Create sensor data object
            sensor_data = PassiveSonarSensorData(
                raw_signals=sensor_signals,
                beamformed_data=beamformed_data,
                timestamp=timestamp,
            )

            yield timestamp, {sensor_data}

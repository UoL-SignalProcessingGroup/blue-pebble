"""Acoustic sensor simulation module."""

from collections.abc import Iterator
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator
from stonesoup.types.sensordata import SensorData

from ..models.propagation import AcousticPropagationModel
from ..platform import TowedArrayPlatform
from ..signal.ambient import AmbientNoise
from ..signal.base import Signal
from ..sigproc.beamformer import Beamformer, SteeringCalculator


class PassiveSonarArraySimulator(SensorSimulator):
    """Stone Soup sensor simulator for passive sonar arrays.

    Simulates acoustic sensor data by generating signals from targets, applying propagation
    effects, adding noise, and performing beamforming. This simulator orchestrates various models
    (propagation, signal, noise) and a beamformer to produce realistic ``PassiveSonarSensorData``.

    Attributes
    ----------
    platform : TowedArrayPlatform
        The towed array platform providing geometry.
    propagation_model : AcousticPropagationModel
        Model for acoustic propagation.
    signal_model : Signal
        Model for generating target signals.
    noise_model : AmbientNoise
        Model for generating ambient noise.
    beamformer : Beamformer
        The beamforming algorithm to apply.
    steering_calculator : SteeringCalculator
        Calculator for steering delays.
    ground_truth_paths : list
        A list of ``GroundTruthPath`` objects representing targets.

    """

    platform = Property(TowedArrayPlatform, doc="Towed array platform")
    propagation_model = Property(AcousticPropagationModel, doc="Acoustic propagation model")
    signal_model = Property(Signal, doc="Acoustic signal model")
    noise_model = Property(AmbientNoise, doc="Noise model")
    beamformer = Property(Beamformer, doc="Beamforming algorithm")
    steering_calculator = Property(SteeringCalculator, doc="Steering calculator")
    ground_truth_paths = Property(list, default=[], doc="List of GroundTruthPath objects")

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate sensor data for each timestamp in the platform's trajectory.

        This generator iterates through all unique timestamps defined in the platform's movement
        controller, yielding a set of sensor data for each point in time.

        Yields
        ------
        tuple
            A tuple containing the timestamp and a set of ``PassiveSonarSensorData`` objects for
            that timestamp.

        """
        all_timestamps = sorted(
            list(set(state.timestamp for state in self.platform.movement_controller.states))
        )

        # Generate sensor data for each timestamp
        for timestamp in all_timestamps:
            sensor_data = self._generate_sensor_data_at(timestamp)
            sensor_data_set = {sensor_data} if sensor_data else set()
            yield timestamp, sensor_data_set

    def _generate_sensor_data_at(self, timestamp) -> PassiveSonarSensorData | None:
        """Generate a single snapshot of sensor data at a specific timestamp.

        This method performs the core simulation steps for a single moment in time. It generates
        signals for all active targets, sums them, adds ambient noise, and then processes the
        result through a beamformer.

        Parameters
        ----------
        timestamp  : datetime
            The timestamp for which to generate data.

        Returns
        -------
        PassiveSonarSensorData | None
            A data object containing the raw signals and beamformed output, or ``None`` if no
            platform state exists at the specified timestamp.

        """
        platform = self.platform.get_platform_state_at(timestamp)

        # Get sensor positions at this timestamp
        num_sensors = self.platform.num_sensors
        num_samples = self.signal_model.num_samples

        # Initialise combined signal array
        sensor_signals = np.zeros((num_sensors, num_samples), dtype=np.complex128)

        # Generate signals from all targets present at this timestamp
        ground_truth_paths = self.ground_truth_paths or []
        for path in ground_truth_paths:
            # Find target state at this timestamp
            target_state = None
            for state in path:
                if state.timestamp == timestamp:
                    target_state = state
                    break

            if target_state is None:
                continue

            # Calculate propagation effects
            tloss_db, prop_time_s = self.propagation_model.propagate(platform, target_state)

            # Calculate sensor delays
            sensor_delays_s = self.propagation_model.compute_sensor_delays(platform, target_state)

            # Generate target signal with acoustic properties from metadata
            target_signal = self.signal_model.generate(
                target_state, sensor_delays_s, tloss_db, prop_time_s
            )

            sensor_signals += target_signal

        # Add environmental noise
        if self.noise_model:
            noise = self.noise_model.generate(num_sensors)
            sensor_signals += noise

        # Calculating steering delays
        steering_delays_s = self.steering_calculator.calculate(platform)

        # Apply beamforming
        beamformed_data = self.beamformer.beamform(sensor_signals, steering_delays_s)

        # Create Stone Soup SensorData object
        sensor_data = PassiveSonarSensorData(
            raw_signals=sensor_signals,
            beamformed_data=beamformed_data,
            timestamp=timestamp,
        )

        return sensor_data


class BroadbandPassiveSonarArraySimulator(SensorSimulator):
    """Stone Soup sensor simulator for broadband passive sonar arrays.

    This simulator uses STFT-based frequency-domain propagation for continuous broadband signal
    processing. Unlike the standard PassiveSonarArraySimulator which generates signals
    per-timestep, this simulator:

    1. Generates a long-duration source signal once using BroadbandTonalSignal
    2. Computes STFT of the source signal
    3. At each timestep, runs rtrs propagation to get H(f) for all frequencies
    4. Applies transfer functions to STFT frames
    5. Reconstructs time-domain signals per sensor using overlap-add

    This approach enables time-varying propagation (moving platforms) with continuous
    phase-coherent signals across the full simulation duration.

    Attributes
    ----------
    platform : TowedArrayPlatform
        The towed array platform providing geometry.
    propagation_model : AcousticPropagationModel
        Model for acoustic propagation (must support ``propagate_spectrum``).
    signal_models : list of Signal
        List of broadband signal models for STFT-based generation. Use a single-element list to
        share one signal model across all targets, or provide one Signal per target for unique
        source characteristics.
    noise_model : AmbientNoise, optional
        Model for generating ambient noise.
    beamformer : Beamformer, optional
        The beamforming algorithm to apply.
    steering_calculator : SteeringCalculator, optional
        Calculator for steering delays.
    ground_truth_paths : list
        List of ``GroundTruthPath`` objects representing targets.
    fade_in_ms : float
        Fade-in duration at signal arrival in milliseconds.

    """

    platform = Property(TowedArrayPlatform, doc="Towed array platform")
    propagation_model = Property(
        AcousticPropagationModel,
        doc="Acoustic propagation model (must support propagate_spectrum)",
    )
    signal_models = Property(
        list,
        doc="List of broadband signal models (one per target, or single-element list for all)",
    )
    noise_model = Property(AmbientNoise, default=None, doc="Noise model (optional)")
    beamformer = Property(Beamformer, default=None, doc="Beamforming algorithm (optional)")
    steering_calculator = Property(
        SteeringCalculator, default=None, doc="Steering calculator (optional)"
    )
    ground_truth_paths = Property(list, default=[], doc="List of GroundTruthPath objects")
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
        all_timestamps = sorted(
            list(set(state.timestamp for state in self.platform.movement_controller.states))
        )

        if len(all_timestamps) < 2:
            msg = "Need at least 2 timesteps for broadband processing"
            raise ValueError(msg)

        # Get all target paths
        ground_truth_paths = self.ground_truth_paths or []
        if len(ground_truth_paths) == 0:
            msg = "BroadbandPassiveSonarArraySimulator requires at least one target"
            raise ValueError(msg)

        # Normalize signal_models to list (support single or per-target)
        if len(self.signal_models) == 1:
            # Single signal model: replicate for all targets
            signal_models_list = self.signal_models * len(ground_truth_paths)
        else:
            # List of signal models: one per target
            signal_models_list = self.signal_models
            if len(signal_models_list) != len(ground_truth_paths):
                msg = (
                    f"Number of signal models ({len(signal_models_list)}) must match "
                    f"number of targets ({len(ground_truth_paths)})"
                )
                raise ValueError(msg)

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
        step_duration_s = total_duration_s / n_steps

        num_sensors = self.platform.num_sensors

        # Storage for transfer functions at each timestep for each target
        # Structure: list of dicts, one dict per target containing:
        #   - 'H_list': list of (num_sensors, num_frequencies) per timestep
        #   - 'tdelay_list': list of propagation delays per timestep
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
            tdelay_list = []  # List of propagation delays per timestep

            # Run propagation simulation for each timestep to get H(f)
            for timestamp in all_timestamps:
                # Get platform state
                platform_state = self.platform.get_platform_state_at(timestamp)

                # Get target state at this timestamp
                target_state = None
                for state in target_path:
                    if state.timestamp == timestamp:
                        target_state = state
                        break

                if target_state is None:
                    # Keep histories aligned with the timestep list so interpolation remains
                    # valid when a target is absent at one or more steps.
                    H_list_all.append(
                        np.zeros((num_sensors, len(frequencies)), dtype=np.complex64)
                    )
                    tdelay_list.append(0.0)
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
                tdelay_list.append(prop_time_s)

            # Store this target's data
            targets_data.append(
                {
                    "H_list": H_list_all,
                    "tdelay_list": tdelay_list,
                    "source_stft": target_source_stft,
                }
            )

        # Now reconstruct signals for each sensor using overlap-add
        # For multiple targets, sum contributions in the frequency domain
        receiver_signals = []

        # Reverse sensor order to match beamformer (expects back-to-front)
        for sensor_idx in reversed(range(num_sensors)):
            # Accumulate STFT output from all targets
            STFT_out_total = np.zeros((num_frames, num_freq_bins), dtype=np.complex64)

            # Process each target
            for target_data in targets_data:
                H_list_all = target_data["H_list"]
                target_source_stft = target_data["source_stft"]

                # Extract transfer function history for this sensor and target
                H_sensor_history = [H_list[sensor_idx, :] for H_list in H_list_all]

                # Interpolate H(f) across time for each STFT frame
                STFT_out_target = np.zeros((num_frames, num_freq_bins), dtype=np.complex64)

                for frame_idx in range(num_frames):
                    # Calculate time for this frame (center of frame)
                    frame_time_s = (frame_idx * hop + hop // 2) / signal_models_list[
                        0
                    ].sampling_rate_hz

                    # Find which timestep this frame belongs to
                    step_idx_float = frame_time_s / step_duration_s
                    step_idx = int(np.floor(step_idx_float))

                    # Clamp to valid range
                    if step_idx >= n_steps - 1:
                        step_idx = n_steps - 2

                    # Interpolation weight
                    alpha = step_idx_float - step_idx

                    # Interpolate H(f) between timesteps
                    H_current = H_sensor_history[step_idx]
                    H_next = H_sensor_history[step_idx + 1]
                    H_interp = H_current * (1 - alpha) + H_next * alpha

                    # Apply transfer function to this target's source STFT
                    STFT_out_target[frame_idx, :] = target_source_stft[frame_idx, :] * H_interp

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
            if self.noise_model:
                # Get actual number of samples in this slice
                # (may differ due to STFT trimming)
                actual_samples = sensor_signals.shape[1]

                # Generate noise with the correct length
                # Temporarily adjust noise model duration to match actual slice length
                original_duration = self.noise_model.duration_s
                actual_duration_s = actual_samples / signal_models_list[0].sampling_rate_hz
                self.noise_model.duration_s = actual_duration_s

                noise = self.noise_model.generate(num_sensors)

                # Restore original duration
                self.noise_model.duration_s = original_duration

                # Ensure noise matches signal length exactly (in case of rounding)
                if noise.shape[1] != actual_samples:
                    if noise.shape[1] > actual_samples:
                        noise = noise[:, :actual_samples]
                    else:
                        # Pad with zeros if needed
                        pad_len = actual_samples - noise.shape[1]
                        noise = np.concatenate(
                            [
                                noise,
                                np.zeros((num_sensors, pad_len), dtype=noise.dtype),
                            ],
                            axis=1,
                        )

                sensor_signals += noise

            # Apply beamforming if provided
            beamformed_data = None
            if self.beamformer and self.steering_calculator:
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

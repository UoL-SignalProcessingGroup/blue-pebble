"""Defines a high-fidelity sonar detector.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.platform.base import Platform
from stonesoup.types.detection import Detection

from nereus.stonesoup.beamformers import Beamformer, SteeringCalculator
from nereus.stonesoup.detection_algorithms import DetectionAlgorithm
from nereus.stonesoup.propagators import AcousticPropagationModel
from nereus.stonesoup.signals import AcousticSignalModel, NoiseModel
from nereus.stonesoup.targets import AcousticTarget


class PassiveSonarProcessingChainDetector:
    """A high-fidelity passive sonar detector.

    Simulates the entire processing chain from signal generation to target detection.
    This component orchestrates a series of models (propagation, signal, noise,
    beamforming, and detection algorithms) to generate detections from ground truths.

    Note: This detector doesn't inherit from Stone Soup base classes to avoid
    property validation issues. It provides the same interface as DetectionReader.
    """

    def __init__(
        self,
        platform: Platform,
        propagation_model: AcousticPropagationModel,
        signal_model: AcousticSignalModel,
        noise_model: NoiseModel,
        steering_calculator: SteeringCalculator,
        beamformer: Beamformer,
        detection_chain: list[DetectionAlgorithm],
        measurement_model: MeasurementModel,
        targets: list[AcousticTarget],
        **kwargs,
    ):
        """Initialize the PassiveSonarProcessingChainDetector.

        Args:
            platform: The platform containing the sensor array.
            propagation_model: Model for acoustic propagation.
            signal_model: Model for acoustic signal generation.
            noise_model: Model for noise generation.
            steering_calculator: Calculator for beam steering.
            beamformer: Beamformer for processing sensor signals.
            detection_chain: List of detection algorithms to apply.
            measurement_model: Model for converting detections to measurements.
            targets: List of acoustic targets to detect.
            **kwargs: Additional keyword arguments passed to parent class.

        """
        super().__init__(**kwargs)
        self.platform = platform
        self.propagation_model = propagation_model
        self.signal_model = signal_model
        self.noise_model = noise_model
        self.steering_calculator = steering_calculator
        self.beamformer = beamformer
        self.detection_chain = detection_chain
        self.measurement_model = measurement_model
        self.targets = targets
        self.propagation_model = propagation_model
        self.signal_model = signal_model
        self.noise_model = noise_model
        self.steering_calculator = steering_calculator
        self.beamformer = beamformer
        self.detection_chain = detection_chain
        self.measurement_model = measurement_model
        self.targets = targets

    def detections_gen(self):
        """Iterate through ground truths to generate detections.

        This method iterates through each time step, simulates the full passive
        sonar processing chain, and yields a set of detections.

        Yields:
            (datetime, set[Detection]): A timestamp and the set of detections
            generated at that timestamp.

        """
        # Get all unique timestamps from the scenario
        all_timestamps = sorted(
            list(set(state.timestamp for state in self.platform.ship.states))
        )

        for time in all_timestamps:
            detections = set()

            # Find platform state at this time
            platform_state = None
            for state in self.platform.ship.states:
                if state.timestamp == time:
                    platform_state = state
                    break

            if platform_state is None:
                yield time, detections
                continue

            current_targets = []
            for target in self.targets:
                # Find target state at this time
                target_state = None
                for state in target.states:
                    if state.timestamp == time:
                        target_state = state
                        break

                if target_state is not None:
                    current_targets.append(target)

            if not current_targets:
                yield time, detections
                continue

            # --- Step 1: Generate Sensor Signals ---
            sensor_signals = self._generate_sensor_signals(
                current_targets, platform_state, time
            )

            # --- Step 2: Beamform the Signals ---
            steering_delays_s = self.steering_calculator.calculate(platform_state)
            beamformed_signals = self.beamformer.beamform(
                sensor_signals, steering_delays_s
            )

            # --- Step 3: Run Detection Chain ---
            if beamformed_signals.size > 0:
                # Calculate power from the time-series beamformed signals
                # Compute magnitude squared for power calculation
                directional_power = np.mean(np.abs(beamformed_signals) ** 2, axis=1)
                directional_power_db = 10 * np.log10(directional_power + 1e-12)
                # Run the detection chain
                raw_detections = self._run_detection_chain(directional_power_db)
            else:
                raw_detections = np.empty((0, 2))

            # --- Step 4: Create Stone Soup Detections ---
            if raw_detections.size > 0:
                for raw_det in raw_detections:
                    # Convert the detected index back to an angle
                    detection_index = raw_det[0]
                    bearing_rad = self.steering_calculator.steering_azimuths_rad[
                        int(detection_index)
                    ]

                    detections.add(
                        Detection(
                            state_vector=[bearing_rad],
                            timestamp=time,
                            measurement_model=self.measurement_model,
                            metadata={"snr_db": raw_det[1]},  # Optionally include SNR
                        )
                    )

            # Yield the detections for this time step (can be an empty set)
            yield time, detections

    def _generate_sensor_signals(self, targets, platform_state, timestamp):
        """Generate the combined noisy signal array."""
        sensors_list = list(self.platform.sensors)  # type: ignore  # Stone Soup property
        num_sensors = len(sensors_list)
        num_samples = self.signal_model.num_samples

        combined_noiseless_signal = np.zeros(
            (num_sensors, num_samples), dtype=np.complex128
        )

        for target in targets:
            # Find target state at this timestamp
            target_state = None
            for state in target.states:
                if state.timestamp == timestamp:
                    target_state = state
                    break

            if target_state is None:
                continue

            tloss_db, prop_time_s = self.propagation_model.propagate(
                platform_state, target_state
            )
            sensor_delays_s = self.propagation_model.compute_sensor_delays(
                platform_state, target_state
            )
            combined_noiseless_signal += self.signal_model.generate(
                target, sensor_delays_s, tloss_db, prop_time_s
            )

        sensor_signals = combined_noiseless_signal
        if self.noise_model:
            noise = self.noise_model.generate(num_sensors)
            sensor_signals += noise

        return sensor_signals

    def _run_detection_chain(self, initial_power_map: np.ndarray) -> np.ndarray:
        """Process a data map through a sequential chain of detection algorithms."""
        if not self.detection_chain:
            return np.empty((0, 2))

        input_data_map = initial_power_map
        final_detections = np.empty((0, 2))

        for algorithm in self.detection_chain:
            current_detections = algorithm.detect(input_data_map)

            if current_detections.size == 0:
                return np.empty((0, 2))  # Chain is broken, no final detections

            final_detections = current_detections

            # Prepare a new masked input for the next algorithm in the chain
            input_data_map = np.full(len(initial_power_map), -np.inf)
            indices = final_detections[:, 0].astype(int)
            values = final_detections[:, 1]
            input_data_map[indices] = values

        return final_detections

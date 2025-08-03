"""Defines a high-fidelity sonar detector.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np
from stonesoup.detector.base import Detector
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.platform.base import Platform
from stonesoup.types.detection import Detection

from nereus.stonesoup.beamformers import (
    Beamformer,
    SteeringCalculator,
    calculate_directional_power,
)
from nereus.stonesoup.detection_algorithms import DetectionAlgorithm
from nereus.stonesoup.propagators import AcousticPropagationModel
from nereus.stonesoup.signals import AcousticSignalModel, NoiseModel


class SonarProcessingChainDetector(Detector):
    """A high-fidelity sonar detector.

    Simulates the entire processing chain from signal generation to target detection.
    This component orchestrates a series of models (propagation, signal, noise,
    beamforming, and detection algorithms) to generate detections from ground truths.
    """

    # --- Component Properties ---
    platform: Platform
    propagation_model: AcousticPropagationModel
    signal_model: AcousticSignalModel
    noise_model: NoiseModel
    steering_calculator: SteeringCalculator
    beamformer: Beamformer
    detection_chain: list[DetectionAlgorithm]
    _measurement_model: MeasurementModel

    @property
    def measurement_model(self) -> MeasurementModel:
        """The measurement model used to create formal Detection objects."""
        return self._measurement_model

    def detections_gen(self, ground_truths, **kwargs):
        """Iterate through ground truths to generate detections.

        This method iterates through each time step, simulates the full passive
        sonar processing chain, and yields a set of detections.

        Args:
            ground_truths (GroundTruthPath): A container of ground truth states.
            **kwargs: Additional keyword arguments for flexibility.

        Yields:
            (datetime, set[Detection]): A timestamp and the set of detections
            generated at that timestamp.

        """
        for time, truths in ground_truths.items():
            detections = set()
            platform_state = self.platform.get_state(time)

            # --- Step 1: Generate Sensor Signals ---
            sensor_signals = self._generate_sensor_signals(truths, platform_state)

            # --- Step 2: Beamform the Signals ---
            steering_delays_s = self.steering_calculator.calculate(platform_state)
            beamformed_signals = self.beamformer.beamform(
                sensor_signals, steering_delays_s
            )

            # --- Step 3: Run Detection Chain ---
            if beamformed_signals.size > 0:
                # Calculate power from the time-series beamformed signals
                directional_power_db = 10 * np.log10(
                    calculate_directional_power(beamformed_signals) + 1e-12
                )
                # Run the detection chain
                raw_detections = self._run_detection_chain(directional_power_db)
            else:
                raw_detections = np.array([])

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

    def _generate_sensor_signals(self, truths, platform_state):
        """Generate the combined noisy signal array."""
        num_sensors = len(self.platform.sensors)
        num_samples = self.signal_model.num_samples

        combined_noiseless_signal = np.zeros(
            (num_sensors, num_samples), dtype=np.complex128
        )

        for truth in truths:
            tloss_db, prop_time_s = self.propagation_model.propagate(
                platform_state, truth
            )
            sensor_delays_s = self.propagation_model.compute_sensor_delays(
                platform_state, truth
            )
            combined_noiseless_signal += self.signal_model.generate(
                truth, sensor_delays_s, tloss_db, prop_time_s
            )

        sensor_signals = combined_noiseless_signal
        if self.noise_model:
            noise = self.noise_model.generate(num_sensors)
            sensor_signals += noise

        return sensor_signals

    def _run_detection_chain(self, initial_power_map: np.ndarray) -> np.ndarray:
        """Process a data map through a sequential chain of detection algorithms."""
        if not self.detection_chain:
            return np.array([])

        input_data_map = initial_power_map
        final_detections = np.array([])

        for algorithm in self.detection_chain:
            current_detections = algorithm.detect(input_data_map)

            if current_detections.size == 0:
                return np.array([])  # Chain is broken, no final detections

            final_detections = current_detections

            # Prepare a new masked input for the next algorithm in the chain
            input_data_map = np.full(len(initial_power_map), -np.inf)
            indices = final_detections[:, 0].astype(int)
            values = final_detections[:, 1]
            input_data_map[indices] = values

        return final_detections

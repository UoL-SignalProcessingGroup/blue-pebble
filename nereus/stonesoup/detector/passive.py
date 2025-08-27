"""Defines a passive sonar detector that processes beamformed sensor data."""

from collections.abc import Generator

import numpy as np
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection

from nereus.stonesoup.detector import DetectionAlgorithm


class PassiveSonarDetector(DetectionReader):
    """A passive sonar detector that processes beamformed sensor data.

    This detector takes PassiveSonarSensorData as input, extracts the
    beamformed power map, and runs a chain of detection algorithms to
    find targets.
    """

    detection_chain = Property(
        list[DetectionAlgorithm],
        doc="A list of detection algorithms to apply sequentially.",
    )
    sensor_data_gen = Property(
        Generator, doc="Generator that yields PassiveSonarSensorData objects"
    )
    steering_azimuths_rad = Property(
        np.ndarray,
        default=None,
        doc="Array of steering azimuth angles in radians. If not provided, "
        "detection indices will be used as bearing values "
        "(for backward compatibility).",
    )

    def __init__(self, *args, **kwargs):
        """Initialise the passive sonar detector."""
        super().__init__(*args, **kwargs)
        self._snr_history = []

    @property
    def snr_history(self):
        """Get the recorded SNR history as a 2D numpy array.

        Returns:
            np.ndarray: Shape (num_timesteps, num_beams) containing SNR values

        """
        if not self._snr_history:
            return np.array([])
        return np.array(self._snr_history)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        """Generate detections from sensor data.

        Yields:
            tuple: (timestamp, set of Detection objects)

        """
        for timestamp, sensor_data_set in self.sensor_data_gen:
            detections = set()

            # Process each sensor data object in the set
            for sensor_data in sensor_data_set:
                # Extract the beamformed data from the sensor data
                beamformed_data = sensor_data.beamformed_data

                if beamformed_data.size == 0:
                    continue

                directional_power = np.mean(np.abs(beamformed_data) ** 2, axis=1)

                # Estimate post-beamforming noise floor
                noise_power_estimate = np.percentile(directional_power, 25)

                # Calculate signal power
                signal_power = directional_power - noise_power_estimate

                # Proper SNR calculation
                epsilon = np.finfo(float).eps
                snr = 10 * np.log10(
                    np.maximum(signal_power, 0) / (noise_power_estimate + epsilon)
                    + epsilon
                )

                # Run the detection chain on the SNR map
                raw_detections = self._run_detection_chain(snr)

                # Create Stone Soup Detections from the raw results
                if raw_detections.size > 0:
                    for raw_det in raw_detections:
                        detection_index = int(raw_det[0])

                        # Convert detection index to bearing angle in radians
                        if self.steering_azimuths_rad is not None:
                            # Use the actual steering azimuth angle
                            bearing_rad = self.steering_azimuths_rad[detection_index]
                        else:
                            # Backward compatibility: use index as bearing value
                            bearing_rad = detection_index

                        detections.add(
                            Detection(
                                state_vector=[[bearing_rad]],
                                timestamp=sensor_data.timestamp,
                                metadata={"snr_db": raw_det[1]},
                            )
                        )

            self._snr_history.append(snr)

            yield timestamp, detections

    def _run_detection_chain(self, initial_snr_map: np.ndarray) -> np.ndarray:
        """Process a data map through a sequential chain of detection algorithms."""
        if not self.detection_chain:
            return np.empty((0, 2))

        input_data_map = initial_snr_map
        final_detections = np.empty((0, 2))

        for algorithm in self.detection_chain:
            current_detections = algorithm.detect(input_data_map)

            if current_detections.size == 0:
                return np.empty((0, 2))

            final_detections = current_detections

            input_data_map = np.full(len(initial_snr_map), -np.inf)
            indices = final_detections[:, 0].astype(int)
            values = final_detections[:, 1]
            input_data_map[indices] = values

        return final_detections

"""Defines a passive sonar detector that processes beamformed sensor data."""

from collections.abc import Generator

import numpy as np
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection

from nereus.detector import DetectionAlgorithm

from tqdm import tqdm


class PassiveSonarDetector(DetectionReader):
    """A passive sonar detector that processes beamformed sensor data.

    This detector takes ``PassiveSonarSensorData`` as input, extracts the
    beamformed power map, calculates the Signal-to-Noise Ratio (SNR) for each
    beam, and then runs a chain of detection algorithms to find targets.

    The SNR is calculated by estimating noise power as the minimum power
    observed across all beams and assuming the remaining power is signal.
    Detections are generated with bearing information derived from the steering
    azimuths.

    Attributes:
        detection_chain (list[DetectionAlgorithm]): A list of detection
            algorithms to apply sequentially to the SNR map.
        sensor_data_gen (Generator): A generator that yields
            ``PassiveSonarSensorData`` objects.
        steering_azimuths_rad (np.ndarray): An array of steering azimuth
            angles in radians, corresponding to the beams.
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
        doc="Array of steering azimuth angles in radians.",
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
    def detections_gen(self, progress_bar: bool = False):
        """Generate detections from sensor data.

        This generator iterates through the `sensor_data_gen`, processes each
        `PassiveSonarSensorData` object to calculate an SNR map, and applies
        the `detection_chain` to identify detections.

        Yields:
            tuple: A tuple containing the timestamp and a set of `Detection`
            objects for that timestamp.

        """
        sensor_data_iterator = self.sensor_data_gen
        if progress_bar:
            sensor_data_iterator = tqdm(sensor_data_iterator, desc="Generating Detections")
        
        for timestamp, sensor_data_set in sensor_data_iterator:
            detections = set()

            # Process each sensor data object in the set
            for sensor_data in sensor_data_set:
                # Extract the beamformed data from the sensor data
                beamformed_data = sensor_data.beamformed_data

                if beamformed_data.size == 0:
                    continue
                
                # Calculate directional power for each beam
                directional_power = np.mean(np.abs(beamformed_data) ** 2, axis=1)

                # Estimate noise power as the 10th percentile of directional power
                # More stable than minimum, avoids outliers and division by zero
                noise_power_estimate = np.percentile(directional_power, 10)

                # Calculate SNR
                epsilon = np.finfo(float).eps
                snr = 10 * np.log10((directional_power + epsilon) / (noise_power_estimate + epsilon))

                # Run the detection chain on the SNR map
                raw_detections = self._run_detection_chain(snr)

                # Create Stone Soup Detections from the raw results
                if raw_detections.size > 0:
                    for raw_det in raw_detections:
                        detection_index = int(raw_det[0])

                        # Convert detection index to bearing angle in radians
                        bearing_rad = self.steering_azimuths_rad[detection_index]

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
        """Process a data map through a sequential chain of detection algorithms.

        This method applies each algorithm in the `detection_chain` in order.
        The output of one algorithm becomes the input for the next. The input
        to subsequent algorithms is a sparse map containing only the values of
        the detections from the previous stage.

        Args:
            initial_snr_map (np.ndarray): The initial 1D data map (e.g., SNR)
                to be processed.

        Returns:
            np.ndarray: A 2D array of final detections, where each row is
            [index, value]. Returns an empty array if no detections are found
            at any stage.
        """
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

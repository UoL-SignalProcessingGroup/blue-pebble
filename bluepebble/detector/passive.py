"""Defines a passive sonar detector that processes beamformed sensor data."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection
from tqdm import tqdm

if TYPE_CHECKING:
    from ..types.sensordata import PassiveSonarSensorData
    from .algorithms import DetectionAlgorithm

FloatArray: TypeAlias = NDArray[np.float64]
DetectionArray: TypeAlias = NDArray[np.float64]
SensorDataStep: TypeAlias = tuple[datetime, Iterable["PassiveSonarSensorData"]]
DetectionBatch: TypeAlias = tuple[datetime, set[Detection]]


class PassiveSonarDetector(DetectionReader):
    """A passive sonar detector that processes beamformed sensor data.

    This detector takes ``PassiveSonarSensorData`` as input, extracts the beamformed power map,
    calculates the Signal-to-Noise Ratio (SNR) for each beam, and then runs a chain of detection
    algorithms to find targets.

    The SNR is calculated by estimating noise power as the 10th-percentile of directional power
    (robust to outliers). Detections are produced with bearing values derived from the provided
    steering azimuths.

    Attributes
    ----------
    detection_chain : list[DetectionAlgorithm]
        A list of detection algorithms to apply sequentially to the SNR map.
    sensor_data_gen : Generator[SensorDataStep, None, None]
        Generator yielding sensor-data batches.
    steering_azimuths_rad : FloatArray
        An array of steering azimuth angles in radians corresponding to the beams.

    """

    detection_chain: list[DetectionAlgorithm] = Property(
        doc="A list of detection algorithms to apply sequentially.",
    )
    sensor_data_gen: Generator[SensorDataStep, None, None] = Property(
        doc="Generator that yields PassiveSonarSensorData objects"
    )
    steering_azimuths_rad: FloatArray = Property(
        doc="Array of steering azimuth angles in radians.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the passive sonar detector."""
        super().__init__(*args, **kwargs)
        self._snr_history: list[FloatArray] = []

    @property
    def snr_history(self) -> FloatArray:
        """Recorded SNR history.

        Returns
        -------
        FloatArray
            Array of shape (num_timesteps, num_beams) containing SNR values. If no history is
            available an empty array is returned.

        """
        if not self._snr_history:
            return np.array([], dtype=np.float64)
        return np.asarray(self._snr_history, dtype=np.float64)

    @BufferedGenerator.generator_method
    def detections_gen(
        self,
        progress_bar: bool = False,
        total_timesteps: int | None = None,
        beamformer_output_type: str = "snr_percentile",
        snr_percentile_val: int = 10,
    ) -> Generator[DetectionBatch, None, None]:
        """Generate detections from sensor data.

        The generator iterates through ``sensor_data_gen``, computes an SNR map for each beamformed
        frame, runs the configured ``detection_chain`` and yields Stone Soup ``Detection`` objects
        (bearing-only measurements).

        Parameters
        ----------
        progress_bar : bool, optional
            If True, wrap the input generator with a progress bar (default is False).
        total_timesteps : int, optional
            Total number of timesteps for the progress bar. Required if `progress_bar` is True.
        beamformer_output_type : str, optional
            Type of beamformer output to use for detection. Options are "snr_percentile" (default),
            "log_power", "power", or "median_power".
        snr_percentile_val : int, optional
            Percentile to use for noise power estimation when calculating SNR (default is 10).

        Yields
        ------
        tuple
            A tuple of ``(timestamp, set[Detection])`` for each processed timestep.

        """
        sensor_data_iterator: Iterable[SensorDataStep] = self.sensor_data_gen
        if progress_bar:
            sensor_data_iterator = tqdm(
                sensor_data_iterator, desc="Generating Detections", total=total_timesteps
            )

        for timestamp, sensor_data_set in sensor_data_iterator:
            detections: set[Detection] = set()
            snr: FloatArray = np.array([], dtype=np.float64)

            # Process each sensor data object in the set
            for sensor_data in sensor_data_set:
                # Extract the beamformed data from the sensor data
                beamformed_data = sensor_data.beamformed_data

                if beamformed_data is None or beamformed_data.size == 0:
                    continue

                if beamformer_output_type == "snr_percentile":
                    # Calculate directional power for each beam
                    directional_power = np.mean(np.abs(beamformed_data) ** 2, axis=1)

                    # Estimate noise power as the 10th percentile of directional power
                    # More stable than minimum, avoids outliers and division by zero
                    noise_power_estimate = np.percentile(directional_power, snr_percentile_val)

                    # Calculate SNR
                    epsilon = np.finfo(float).eps
                    snr = 10 * np.log10(
                        (directional_power + epsilon) / (noise_power_estimate + epsilon)
                    )
                elif beamformer_output_type == "median_power":
                    directional_power = np.mean(np.abs(beamformed_data) ** 2, axis=1)
                    noise_power_estimate = np.median(directional_power)
                    epsilon = np.finfo(float).eps
                    snr = 10 * np.log10(
                        (directional_power + epsilon) / (noise_power_estimate + epsilon)
                    )
                elif beamformer_output_type == "log_power":
                    snr = 10 * np.log10(np.mean(np.abs(beamformed_data) ** 2, axis=1))
                elif beamformer_output_type == "power":
                    snr = np.mean(np.abs(beamformed_data) ** 2, axis=1)
                else:
                    raise ValueError(
                        f"Unsupported beamformer_output_type: {beamformer_output_type}"
                    )

                # Run the detection chain on the SNR map
                raw_detections: DetectionArray = self._run_detection_chain(snr)

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

    def _run_detection_chain(self, initial_snr_map: ArrayLike) -> DetectionArray:
        """Process a data map through a sequential chain of detection algorithms.

        Each algorithm in ``detection_chain`` is applied in sequence; the set of detections
        produced by one stage is converted to a sparse input map for the next stage (non-detected
        indices set to -inf).

        Parameters
        ----------
        initial_snr_map : ArrayLike
            The initial 1D data map (for example, SNR in dB) to be processed.

        Returns
        -------
        DetectionArray
            A 2D array of final detections where each row is ``[index, value]``. Returns an empty
            array if no detections are found at any stage.

        """
        initial_snr_map_array = np.asarray(initial_snr_map, dtype=np.float64)
        if not self.detection_chain:
            return np.empty((0, 2), dtype=np.float64)

        input_data_map = initial_snr_map_array
        final_detections = np.empty((0, 2), dtype=np.float64)

        for algorithm in self.detection_chain:
            current_detections = algorithm.detect(input_data_map)

            if current_detections.size == 0:
                return np.empty((0, 2), dtype=np.float64)

            final_detections = current_detections

            input_data_map = np.full(len(initial_snr_map_array), -np.inf, dtype=np.float64)
            indices = final_detections[:, 0].astype(int)
            values = final_detections[:, 1]
            input_data_map[indices] = values

        return np.asarray(final_detections, dtype=np.float64)

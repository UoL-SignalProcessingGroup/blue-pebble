"""Acoustic sensor simulation module."""

from collections.abc import Iterator
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator
from stonesoup.types.sensordata import SensorData

from nereus.stonesoup.models.propagation import AcousticPropagationModel
from nereus.stonesoup.platform import TowedArrayPlatform
from nereus.stonesoup.sigproc.beamformer import Beamformer, SteeringCalculator
from nereus.stonesoup.sigproc.signal import AcousticSignalModel, NoiseModel


class PassiveSonarSensorData(SensorData):
    """Custom sensor data for passive sonar arrays.

    Contains acoustic sensor array data including raw signals,
    beamformed data, and associated metadata.
    """

    raw_signals = Property(np.ndarray, doc="Raw acoustic signals from sensor array")
    beamformed_data = Property(np.ndarray, doc="Processed beamformed output")
    timestamp = Property(datetime, doc="Timestamp of the sensor data")


class PassiveSonarArraySimulator(SensorSimulator):
    """Stone Soup sensor simulator for passive sonar arrays.

    Simulates acoustic sensor data by generating signals from targets,
    applying propagation effects, adding noise, and performing beamforming.
    """

    platform = Property(TowedArrayPlatform, doc="Towed array platform")
    propagation_model = Property(
        AcousticPropagationModel, doc="Acoustic propagation model"
    )
    signal_model = Property(AcousticSignalModel, doc="Acoustic signal model")
    noise_model = Property(NoiseModel, doc="Noise model")
    beamformer = Property(Beamformer, doc="Beamforming algorithm")
    steering_calculator = Property(SteeringCalculator, doc="Steering calculator")
    ground_truth_paths = Property(
        list, default=[], doc="List of GroundTruthPath objects"
    )

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate sensor data over time.

        Yields:
            tuple: (timestamp, set of sensor data) for each time step

        """
        all_timestamps = sorted(
            list(
                set(
                    state.timestamp
                    for state in self.platform.movement_controller.states
                )
            )
        )

        # Generate sensor data for each timestamp
        for timestamp in all_timestamps:
            sensor_data = self._generate_sensor_data_at(timestamp)
            sensor_data_set = {sensor_data} if sensor_data else set()
            yield timestamp, sensor_data_set

    def _generate_sensor_data_at(self, timestamp) -> PassiveSonarSensorData | None:
        """Generate sensor data at a specific timestamp.

        Args:
            timestamp: The timestamp to generate data for

        Returns:
            PassiveSonarSensorData: Sensor data containing beamformed signals
                                    and metadata, or None if no platform state

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
            tloss_db, prop_time_s = self.propagation_model.propagate(
                platform, target_state
            )

            # Calculate sensor delays
            sensor_delays_s = self.propagation_model.compute_sensor_delays(
                platform, target_state
            )

            # Generate target signal with acoustic properties from metadata
            target_signal = self.signal_model.generate(
                target_state, sensor_delays_s, tloss_db, prop_time_s
            )

            # print(np.max(np.abs(target_signal)))

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

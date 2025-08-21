from collections.abc import Iterator
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator
from stonesoup.types.sensordata import SensorData

from nereus.stonesoup.beamformers import Beamformer
from nereus.stonesoup.movable import TowedArrayPlatform
from nereus.stonesoup.propagators import AcousticPropagationModel
from nereus.stonesoup.signals import AcousticSignalModel, NoiseModel


class PassiveSonarSensorData(SensorData):
    """Custom sensor data for passive sonar arrays.

    Contains acoustic sensor array data including raw signals,
    beamformed data, and associated metadata.
    """

    raw_signals = Property(np.ndarray, doc="Raw acoustic signals from sensor array")
    beamformed_data = Property(np.ndarray, doc="Processed beamformed output")
    platform_state = Property(object, doc="Current platform state")
    sensor_positions = Property(list, doc="Positions of all sensors")
    targets_metadata = Property(list, doc="Metadata about detected targets")
    num_sensors = Property(int, doc="Number of sensors in array")
    num_samples = Property(int, doc="Number of signal samples")


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
    ground_truth_paths = Property(
        list, default=None, doc="List of GroundTruthPath objects"
    )

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate sensor data over time.

        Yields:
            tuple: (timestamp, set of sensor data) for each time step

        """
        # Get all unique timestamps from ground truth paths and platform
        timestamps = set()

        # Add platform timestamps
        for state in self.platform.ship.states:
            timestamps.add(state.timestamp)

        # Add target timestamps
        ground_truth_paths = self.ground_truth_paths or []
        for path in ground_truth_paths:
            for state in path:
                timestamps.add(state.timestamp)

        # Sort timestamps
        sorted_timestamps = sorted(timestamps)

        # Generate sensor data for each timestamp
        for timestamp in sorted_timestamps:
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
        # Get platform state at this timestamp
        platform_state = None
        for state in self.platform.ship.states:
            if state.timestamp == timestamp:
                platform_state = state
                break

        if platform_state is None:
            return None

        # Get sensor positions at this timestamp
        sensors_list = list(self.platform.sensors)
        num_sensors = len(sensors_list)
        num_samples = self.signal_model.num_samples

        # Initialize combined signal array
        combined_signal = np.zeros((num_sensors, num_samples), dtype=np.complex128)

        # Generate signals from all targets present at this timestamp
        target_metadata = []
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
                platform_state, target_state
            )

            # Calculate sensor delays
            sensor_delays_s = self.propagation_model.compute_sensor_delays(
                platform_state, target_state
            )

            # Generate target signal with acoustic properties from metadata
            target_signal = self.signal_model.generate_from_metadata(
                target_state.metadata, sensor_delays_s, tloss_db, prop_time_s
            )

            combined_signal += target_signal

            # Store target metadata for this timestamp
            target_metadata.append(
                {
                    "position": target_state.state_vector[[0, 2, 4]],
                    "amplitudes": target_state.metadata.get("amplitudes_upa", []),
                    "frequencies": target_state.metadata.get("frequencies_hz", []),
                    "phases": target_state.metadata.get("phases_rad", []),
                    "transmission_loss_db": tloss_db,
                    "propagation_time_s": prop_time_s,
                }
            )

        # Add environmental noise
        if self.noise_model:
            noise = self.noise_model.generate(num_sensors)
            combined_signal += noise

        # Apply beamforming
        beamformed_data = self.beamformer.process(
            combined_signal, sensors_list, timestamp
        )

        # Create Stone Soup SensorData object
        sensor_data = PassiveSonarSensorData(
            raw_signals=combined_signal,
            beamformed_data=beamformed_data,
            platform_state=platform_state,
            sensor_positions=[sensor.get_state(timestamp) for sensor in sensors_list],
            targets_metadata=target_metadata,
            num_sensors=num_sensors,
            num_samples=num_samples,
            timestamp=timestamp,
        )

        return sensor_data

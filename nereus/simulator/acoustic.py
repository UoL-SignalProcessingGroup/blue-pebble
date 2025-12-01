"""Acoustic sensor simulation module."""

from collections.abc import Iterator
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator
from stonesoup.types.sensordata import SensorData

from nereus.models.propagation import AcousticPropagationModel
from nereus.platform import TowedArrayPlatform
from nereus.signal.ambient import AmbientNoise
from nereus.signal.base import Signal
from nereus.sigproc.beamformer import Beamformer, SteeringCalculator


class PassiveSonarSensorData(SensorData):
    """Custom sensor data for passive sonar arrays.

    This class extends Stone Soup's ``SensorData`` to include data specific
    to passive sonar simulation. It holds the raw time-series signals from
    each sensor, the final beamformed power map, and the timestamp of the
    data snapshot.
    """

    raw_signals = Property(np.ndarray, doc="Raw acoustic signals from sensor array")
    beamformed_data = Property(np.ndarray, doc="Processed beamformed output")
    timestamp = Property(datetime, doc="Timestamp of the sensor data")


class PassiveSonarArraySimulator(SensorSimulator):
    """Stone Soup sensor simulator for passive sonar arrays.

    Simulates acoustic sensor data by generating signals from targets,
    applying propagation effects, adding noise, and performing beamforming.
    This simulator orchestrates various models (propagation, signal, noise)
    and a beamformer to produce realistic ``PassiveSonarSensorData``.

    Attributes:
        platform (TowedArrayPlatform): The towed array platform providing geometry.
        propagation_model (AcousticPropagationModel): Model for acoustic propagation.
        signal_model (Signal): Model for generating target signals.
        noise_model (AmbientNoise): Model for generating ambient noise.
        beamformer (Beamformer): The beamforming algorithm to apply.
        steering_calculator (SteeringCalculator): Calculator for steering delays.
        ground_truth_paths (list): A list of ``GroundTruthPath`` objects representing
            targets.

    """

    platform = Property(TowedArrayPlatform, doc="Towed array platform")
    propagation_model = Property(
        AcousticPropagationModel, doc="Acoustic propagation model"
    )
    signal_model = Property(Signal, doc="Acoustic signal model")
    noise_model = Property(AmbientNoise, doc="Noise model")
    beamformer = Property(Beamformer, doc="Beamforming algorithm")
    steering_calculator = Property(SteeringCalculator, doc="Steering calculator")
    ground_truth_paths = Property(
        list, default=[], doc="List of GroundTruthPath objects"
    )

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate sensor data for each timestamp in the platform's trajectory.

        This generator iterates through all unique timestamps defined in the
        platform's movement controller, yielding a set of sensor data for each
        point in time.

        Yields:
            tuple: A tuple containing the timestamp and a set of
            ``PassiveSonarSensorData`` objects for that timestamp.

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
        """Generate a single snapshot of sensor data at a specific timestamp.

        This method performs the core simulation steps for a single moment in
        time. It generates signals for all active targets, sums them, adds
        ambient noise, and then processes the result through a beamformer.

        Args:
            timestamp (datetime): The timestamp for which to generate data.

        Returns:
            PassiveSonarSensorData | None: A data object containing the raw
            signals and beamformed output, or ``None`` if no platform state
            exists at the specified timestamp.

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

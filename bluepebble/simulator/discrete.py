"""Acoustic sensor snapshot simulatiors module."""

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
from .sensor_data import PassiveSonarSensorData


class DiscretePassiveSonarArraySimulator(SensorSimulator):
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
    signal_models : list of Signal
        Signal models for target generation. Provide one model to share across all targets, or one
        model per target.
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
    signal_models = Property(
        list,
        doc="List of acoustic signal models (one per target, or single-element list for all)",
    )
    noise_model = Property(AmbientNoise, doc="Noise model")
    beamformer = Property(Beamformer, doc="Beamforming algorithm")
    steering_calculator = Property(SteeringCalculator, doc="Steering calculator")
    ground_truth_paths = Property(list, default=[], doc="List of GroundTruthPath objects")

    def _resolve_signal_models(self, num_targets: int) -> list[Signal]:
        """Resolve signal model mapping for targets.

        Parameters
        ----------
        num_targets : int
            Number of targets to model at the current timestamp.

        Returns
        -------
        list of Signal
            A list of signal models with one entry per target when ``num_targets > 1``.

        """
        models = list(self.signal_models)
        if len(models) == 0:
            msg = "signal_models must contain at least one Signal"
            raise ValueError(msg)

        if num_targets <= 1:
            return [models[0]]

        if len(models) == 1:
            return models * num_targets

        if len(models) != num_targets:
            msg = (
                f"Number of signal models ({len(models)}) must match "
                f"number of targets ({num_targets}), or be 1"
            )
            raise ValueError(msg)

        return models

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
        ground_truth_paths = self.ground_truth_paths or []
        signal_models_list = self._resolve_signal_models(len(ground_truth_paths))
        num_samples = signal_models_list[0].num_samples

        # Initialise combined signal array
        sensor_signals = np.zeros((num_sensors, num_samples), dtype=np.complex128)

        # Generate signals from all targets present at this timestamp
        for target_idx, path in enumerate(ground_truth_paths):
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
            target_signal_model = signal_models_list[target_idx]
            target_signal = target_signal_model.generate(
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
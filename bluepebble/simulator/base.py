"""Base simulator hierarchy and shared utilities for passive-sonar simulations."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator

from ..models.propagation import AcousticPropagationModel
from ..platform import TowedArrayPlatform
from ..signal.ambient import AmbientNoise
from ..sigproc.beamformer import Beamformer, SteeringCalculator
from .sensordata import PassiveSonarSensorData


class PassiveSonarArraySimulatorBase(SensorSimulator):
    """Common base class for passive-sonar array simulators.

    This class centralizes properties and small utility methods shared by both
    discrete-time and broadband simulators.
    """

    platform = Property(TowedArrayPlatform, doc="Towed array platform")
    propagation_model = Property(AcousticPropagationModel, doc="Acoustic propagation model")
    noise_model = Property(AmbientNoise, default=None, doc="Noise model (optional)")
    beamformer = Property(Beamformer, default=None, doc="Beamforming algorithm (optional)")
    steering_calculator = Property(
        SteeringCalculator,
        default=None,
        doc="Steering calculator (optional)",
    )
    ground_truth_paths = Property(list, default=[], doc="List of GroundTruthPath objects")

    def _sorted_timestamps(self) -> list[datetime]:
        """Return sorted unique platform timestamps."""
        return sorted(
            list(set(state.timestamp for state in self.platform.movement_controller.states))
        )

    @staticmethod
    def _target_state_at(target_path, timestamp: datetime):
        """Return target state at ``timestamp`` or ``None`` if absent."""
        for state in target_path:
            if state.timestamp == timestamp:
                return state
        return None

    @staticmethod
    def _resolve_models(models: list, num_targets: int, model_name: str) -> list:
        """Resolve one model per target from shared or per-target model lists."""
        resolved = list(models)
        if len(resolved) == 0:
            msg = f"{model_name} must contain at least one model"
            raise ValueError(msg)

        if num_targets <= 1:
            return [resolved[0]]

        if len(resolved) == 1:
            return resolved * num_targets

        if len(resolved) != num_targets:
            msg = (
                f"Number of {model_name} ({len(resolved)}) must match "
                f"number of targets ({num_targets}), or be 1"
            )
            raise ValueError(msg)

        return resolved

    def _generate_noise(
        self,
        num_sensors: int,
        num_samples: int,
        sampling_rate_hz: float | None = None,
    ) -> np.ndarray | None:
        """Generate noise and force it to match ``(num_sensors, num_samples)``."""
        if not self.noise_model:
            return None

        original_duration = None
        if sampling_rate_hz is not None and hasattr(self.noise_model, "duration_s"):
            original_duration = self.noise_model.duration_s
            self.noise_model.duration_s = num_samples / sampling_rate_hz

        try:
            noise = self.noise_model.generate(num_sensors)
        finally:
            if original_duration is not None:
                self.noise_model.duration_s = original_duration

        if noise.shape[1] > num_samples:
            return noise[:, :num_samples]

        if noise.shape[1] < num_samples:
            pad_len = num_samples - noise.shape[1]
            return np.concatenate(
                [noise, np.zeros((num_sensors, pad_len), dtype=noise.dtype)],
                axis=1,
            )

        return noise

    def _beamform_if_configured(
        self,
        timestamp: datetime,
        sensor_signals: np.ndarray,
    ):
        """Run beamforming when beamformer and steering calculator are configured."""
        if not (self.beamformer and self.steering_calculator):
            return None

        platform_state = self.platform.get_platform_state_at(timestamp)
        steering_delays_s = self.steering_calculator.calculate(platform_state)
        return self.beamformer.beamform(sensor_signals, steering_delays_s)

    @staticmethod
    def _make_sensor_data(
        timestamp: datetime,
        sensor_signals: np.ndarray,
        beamformed_data,
    ) -> PassiveSonarSensorData:
        """Build a passive-sonar sensor-data payload."""
        return PassiveSonarSensorData(
            raw_signals=sensor_signals,
            beamformed_data=beamformed_data,
            timestamp=timestamp,
        )

    @abstractmethod
    def sensor_data_gen(self):
        """Yield timestamped sensor-data snapshots."""


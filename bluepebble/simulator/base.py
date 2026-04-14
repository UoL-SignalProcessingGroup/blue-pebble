"""Base simulator hierarchy and shared utilities for passive-sonar simulations."""

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator

from ..models.hydrophone import HydrophoneModel, evaluate_hydrophone_transfer_functions
from ..models.propagation import AcousticPropagationModel
from ..platform import TowedArrayPlatform
from ..signal.random import RandomSignal
from ..sigproc.beamformer import Beamformer, SteeringCalculator
from ..types.sensordata import PassiveSonarSensorData

if TYPE_CHECKING:
    from stonesoup.types.groundtruth import GroundTruthPath
    from stonesoup.types.state import State

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
SensorBatch: TypeAlias = tuple[datetime, set[PassiveSonarSensorData]]
TModel = TypeVar("TModel")


class PassiveSonarArraySimulatorBase(SensorSimulator):
    """Common base class for passive-sonar array simulators.

    This class centralises properties and utility methods shared by
    discrete-time and continuous broadband simulator implementations.
    """

    platform: TowedArrayPlatform = Property(doc="Towed array platform")
    propagation_model: AcousticPropagationModel = Property(
        doc="Acoustic propagation model",
    )
    noise_model: RandomSignal | None = Property(
        default=None,
        doc="Stochastic signal model (optional)",
    )
    beamformer: Beamformer | None = Property(
        default=None,
        doc="Beamforming algorithm (optional)",
    )
    steering_calculator: SteeringCalculator | None = Property(
        default=None,
        doc="Steering calculator (optional)",
    )
    ground_truth_paths: list["GroundTruthPath"] = Property(
        default=None,
        doc="List of GroundTruthPath objects",
    )
    hydrophone_models: HydrophoneModel | list[HydrophoneModel] | None = Property(
        default=None,
        doc=(
            "Hydrophone receive model(s).  A single HydrophoneModel applies "
            "uniformly to all sensors; a list provides per-sensor models.  "
            "None disables hydrophone processing (backwards compatible)."
        ),
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise simulator state.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``SensorSimulator``.
        **kwargs : object
            Keyword arguments forwarded to ``SensorSimulator``.

        Notes
        -----
        The ``ground_truth_paths`` property is normalised to a mutable list.

        """
        super().__init__(*args, **kwargs)
        if self.ground_truth_paths is None:
            self.ground_truth_paths = []
        else:
            self.ground_truth_paths = list(self.ground_truth_paths)

    def _sorted_timestamps(self) -> list[datetime]:
        """Return unique platform timestamps in ascending order.

        Returns
        -------
        list of datetime
            Sorted timestamps extracted from platform movement states.

        """
        return sorted(
            list(set(state.timestamp for state in self.platform.movement_controller.states))
        )

    @staticmethod
    def _target_state_at(target_path: Iterable["State"], timestamp: datetime) -> "State | None":
        """Return the target state at a requested timestamp.

        Parameters
        ----------
        target_path : Iterable[State]
            Iterable of target states, each expected to provide ``timestamp``.
        timestamp : datetime
            Timestamp to match.

        Returns
        -------
        State or None
            Matching state object when present; otherwise ``None``.

        """
        for state in target_path:
            if state.timestamp == timestamp:
                return state
        return None

    @staticmethod
    def _resolve_models(
        models: Sequence[TModel],
        num_targets: int,
        model_name: str,
    ) -> list[TModel]:
        """Resolve one model instance per target.

        Parameters
        ----------
        models : Sequence
            Input model sequence, containing either one shared model or one model per target.
        num_targets : int
            Number of targets requiring models.
        model_name : str
            Human-readable model label used in validation errors.

        Returns
        -------
        list
            Resolved model list with one entry per target when ``num_targets > 1``.

        Raises
        ------
        ValueError
            If no models are supplied or if the number of models is invalid
            for the target count.

        """
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
    ) -> ComplexArray | None:
        """Generate and shape noise for a sensor snapshot.

        Parameters
        ----------
        num_sensors : int
            Number of sensor channels.
        num_samples : int
            Required number of samples per channel.

        Returns
        -------
        numpy.ndarray or None
            Complex noise array of shape ``(num_sensors, num_samples)``, or ``None`` when no noise
            model is configured.

        """
        if not self.noise_model:
            return None

        noise = self.noise_model.generate(num_sensors, num_samples=num_samples)

        if noise.shape[1] > num_samples:
            return noise[:, :num_samples]

        if noise.shape[1] < num_samples:
            pad_len = num_samples - noise.shape[1]
            return np.concatenate(
                [noise, np.zeros((num_sensors, pad_len), dtype=noise.dtype)],
                axis=1,
            )

        return noise

    def _apply_hydrophone_to_chunk(
        self,
        sensor_signals: ComplexArray,
        sampling_rate_hz: float,
    ) -> ComplexArray:
        """Apply the hydrophone transfer function to a time-domain sensor chunk.

        Converts the chunk to the frequency domain, multiplies by the hydrophone
        transfer function, and converts back.  This ensures ambient noise added
        before this call passes through the same hydrophone response as the signal.

        If no hydrophone model is configured the input is returned unchanged.

        Parameters
        ----------
        sensor_signals : numpy.ndarray
            Complex sensor data, shape ``(num_sensors, num_samples)``, in the
            pressure domain (before hydrophone conversion).
        sampling_rate_hz : float
            Sampling rate in Hz, used to compute the FFT frequency axis.

        Returns
        -------
        numpy.ndarray
            Complex sensor data in the voltage domain, same shape as input.

        """
        if self.hydrophone_models is None:
            return sensor_signals

        num_sensors, num_samples = sensor_signals.shape
        frequencies_hz = np.fft.fftfreq(num_samples, d=1.0 / sampling_rate_hz)
        H = self._evaluate_hydrophone_tf(num_sensors, frequencies_hz)
        if H is None:
            return sensor_signals

        H_c64 = np.asarray(H, dtype=np.complex64)
        s_fft = np.fft.fft(sensor_signals, axis=1)
        return np.fft.ifft(s_fft * H_c64, axis=1).astype(np.complex64)

    def _evaluate_hydrophone_tf(
        self,
        num_sensors: int,
        frequencies_hz: NDArray[np.floating[Any]],
    ) -> ComplexArray | None:
        """Evaluate the hydrophone transfer function matrix if configured.

        Parameters
        ----------
        num_sensors : int
            Number of sensor elements in the array.
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate.

        Returns
        -------
        numpy.ndarray or None
            Complex array of shape ``(num_sensors, num_frequencies)`` when
            hydrophone models are configured; otherwise ``None``.

        """
        if self.hydrophone_models is None:
            return None
        return evaluate_hydrophone_transfer_functions(
            self.hydrophone_models,
            num_sensors,
            frequencies_hz,
        )

    def _beamform_if_configured(
        self,
        timestamp: datetime,
        sensor_signals: ComplexArray,
    ) -> object | None:
        """Run beamforming for one snapshot when configured.

        Parameters
        ----------
        timestamp : datetime
            Snapshot timestamp used to query platform state.
        sensor_signals : numpy.ndarray
            Complex sensor data with shape ``(num_sensors, num_samples)``.

        Returns
        -------
        object or None
            Beamformer output when both beamformer and steering calculator are configured;
            otherwise ``None``.

        """
        if not (self.beamformer and self.steering_calculator):
            return None

        platform_state = self.platform.get_platform_state_at(timestamp)
        steering_delays_s = self.steering_calculator.calculate(platform_state)
        return self.beamformer.beamform(sensor_signals, steering_delays_s)

    @staticmethod
    def _make_sensor_data(
        timestamp: datetime,
        sensor_signals: ComplexArray,
        beamformed_data: object | None,
    ) -> PassiveSonarSensorData:
        """Build a passive-sonar sensor-data payload.

        Parameters
        ----------
        timestamp : datetime
            Snapshot timestamp.
        sensor_signals : numpy.ndarray
            Complex raw sensor signals with shape ``(num_sensors, num_samples)``.
        beamformed_data : object
            Optional beamformer output payload.

        Returns
        -------
        PassiveSonarSensorData
            Stone Soup-compatible sensor-data object.

        """
        return PassiveSonarSensorData(
            raw_signals=sensor_signals,
            beamformed_data=beamformed_data,
            timestamp=timestamp,
        )

    @abstractmethod
    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield timestamped passive-sonar sensor snapshots.

        Yields
        ------
        tuple of (datetime, set of PassiveSonarSensorData)
            Timestamp and one-element sensor-data set for that timestamp.

        """

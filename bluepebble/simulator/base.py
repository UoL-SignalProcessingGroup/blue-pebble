"""Base simulator hierarchy and shared utilities for passive-sonar simulations."""

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.simulator.base import SensorSimulator

from ..models.propagation import AcousticPropagationModel
from ..platform import TowedArrayPlatform
from ..signal.random import RandomSignal
from ..sigproc.base import Beamformer
from ..sigproc.steering import SteeringCalculator
from ..types.sensordata import PassiveSonarSensorData

if TYPE_CHECKING:
    from stonesoup.types.groundtruth import GroundTruthPath
    from stonesoup.types.state import State

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.float64]
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
    def _states_by_timestamp(target_path: Iterable["State"]) -> dict[datetime, "State"]:
        """Index a target's states by timestamp, keeping the first state at each time.

        Built once per run, so each step's lookup does not rescan the path; scanning made the
        total cost grow with the square of the run length.

        Parameters
        ----------
        target_path : Iterable[State]
            Iterable of target states, each expected to provide ``timestamp``.

        Returns
        -------
        dict
            Maps each timestamp to the first state carrying it. Look a step up with ``.get``,
            which gives ``None`` when the target has no state then.

        """
        states: dict[datetime, State] = {}
        for state in target_path:
            states.setdefault(state.timestamp, state)
        return states

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

    def _beamform_if_configured(
        self,
        timestamp: datetime,
        sensor_signals: ComplexArray,
    ) -> tuple[object | None, FloatArray | None]:
        """Run beamforming for one snapshot when configured.

        Parameters
        ----------
        timestamp : datetime
            Snapshot timestamp used to query platform state.
        sensor_signals : numpy.ndarray
            Complex sensor data with shape ``(num_sensors, num_samples)``.

        Returns
        -------
        tuple of (object or None, FloatArray or None)
            The beamformer output and each beam's bearing for this snapshot, both from the
            same platform state, when a beamformer and steering calculator are configured;
            otherwise ``(None, None)``. When the steering calculator has
            ``mirror_half_plane`` set, the output is already expanded to the full steering
            grid, so the halved computation is transparent to callers.

        Raises
        ------
        ValueError
            If the platform has no state at ``timestamp``.

        """
        if not (self.beamformer and self.steering_calculator):
            return None, None

        platform_state = self.platform.get_platform_state_at(timestamp)
        if platform_state is None:
            raise ValueError(
                f"The platform has no state at {timestamp}, so there is no array geometry "
                "to steer with; move the platform through every simulated timestamp first."
            )
        steering_delays_s = self.steering_calculator.calculate(platform_state)
        # Read the same way as mirror_half_plane below, so steering calculators that predate
        # per-scan bearings (custom subclasses, test doubles) still work, without bearings.
        steering_bearings = getattr(self.steering_calculator, "steering_bearings", None)
        bearings = steering_bearings(platform_state) if steering_bearings else None

        # Only pass mirror_plan when mirroring is actually enabled, so beamformers that
        # predate this parameter (custom subclasses, test doubles) are unaffected -- the
        # non-mirrored call shape is exactly what it was before mirror_half_plane existed.
        if getattr(self.steering_calculator, "mirror_half_plane", False):
            mirror_plan = self.steering_calculator.mirror_plan(platform_state)
            output = self.beamformer.beamform(sensor_signals, steering_delays_s, mirror_plan)
        else:
            output = self.beamformer.beamform(sensor_signals, steering_delays_s)
        return output, bearings

    def _band_labels(self) -> list[str] | None:
        """Return the configured beamformer's band labels.

        Returns
        -------
        list of str or None
            One label per band when the beamformer is configured for multiband output;
            ``None`` for single-band beamformers, for those that do not support bands at
            all, and when no beamformer is configured.

        """
        bands = getattr(self.beamformer, "bands", None)
        if not bands:
            return None
        return [band.label for band in bands]

    def _make_sensor_data(
        self,
        timestamp: datetime,
        sensor_signals: ComplexArray,
        beamformed_data: object | None,
        steering_bearings_rad: FloatArray | None = None,
    ) -> PassiveSonarSensorData:
        """Build a passive-sonar sensor-data payload.

        Band labels are taken from the configured beamformer rather than the caller, so a
        multiband beamformer's output arrives downstream already keyed by band.

        Parameters
        ----------
        timestamp : datetime
            Snapshot timestamp.
        sensor_signals : numpy.ndarray
            Complex raw sensor signals with shape ``(num_sensors, num_samples)``.
        beamformed_data : object
            Optional beamformer output payload.
        steering_bearings_rad : FloatArray, optional
            Each beam's bearing for this snapshot, from :meth:`_beamform_if_configured`.

        Returns
        -------
        PassiveSonarSensorData
            Stone Soup-compatible sensor-data object.

        """
        return PassiveSonarSensorData(
            raw_signals=sensor_signals,
            beamformed_data=beamformed_data,
            timestamp=timestamp,
            band_labels=self._band_labels() if beamformed_data is not None else None,
            steering_bearings_rad=steering_bearings_rad,
        )

    @abstractmethod
    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield timestamped passive-sonar sensor snapshots.

        Yields
        ------
        tuple of (datetime, set of PassiveSonarSensorData)
            Timestamp and one-element sensor-data set for that timestamp.

        """

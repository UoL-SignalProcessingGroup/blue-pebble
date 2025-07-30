"""Provide implementations for track initiation.

This module is a derivative work of Stone Soup, available at
https://github.com/dstl/StoneSoup.

The original work is licensed under the MIT License.
© Crown Copyright 2017-2025 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2025 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2025 University of Liverpool UK
© Copyright 2020-2025 Fraunhofer FKIE
© Copyright 2020-2025 John Hiles
© Copyright 2020-2025 Riskaware Ltd
© Copyright 2021-2025 Roke Manor Research Ltd UK
© Copyright 2023-2025 Loughborough University UK
© Copyright 2025 Joshua J. Wakefield
"""  # noqa: E501

from abc import ABC, abstractmethod
from datetime import timedelta

import numpy as np

from nereus.association.hypothesisers import Hypothesiser
from nereus.filters.predictors import Predictor
from nereus.filters.updaters import Updater
from nereus.management.deleters import Deleter
from nereus.types.arrays import StateVector
from nereus.types.detections import Detection
from nereus.types.states import GaussianState, ParticleState, Track


class Initiator(ABC):
    """Abstract base class for track initiators."""

    @abstractmethod
    def initiate(self, detections: set[Detection]) -> set[Track]:
        """Initiate tracks from a set of detections.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class _SinglePointInitiator(Initiator, ABC):
    """An internal base class for initiators that create tracks from single detections.

    This class provides a common method for converting a detection's measurement
    into a state vector.
    """

    def _detection_to_vector(self, detection: Detection) -> StateVector:
        """Convert a detection into a state vector using its measurement model."""
        measurement_model = detection.measurement_model
        state_vector = measurement_model.inverse_function(detection)

        # Pad the state vector with zeros for velocity components if needed
        if measurement_model.ndim == 4:
            state_vector = StateVector([state_vector[0, 0], 0, state_vector[1, 0], 0])
        elif measurement_model.ndim == 2:
            state_vector = StateVector([state_vector[0, 0], 0])
        return state_vector


class GaussianStateInitiator(_SinglePointInitiator):
    """Initialises a new `GaussianState` track from a single detection."""

    def __init__(self, prior_covariance: np.ndarray):
        """Initialise the Gaussian state initiator.

        Args:
            prior_covariance (np.ndarray): The covariance matrix to assign to
                the new state.

        """
        self.prior_covariance = prior_covariance

    def initiate(self, detections: set[Detection]) -> set[Track]:
        """Initiate a new `GaussianState` track for each detection.

        Args:
            detections (set[Detection]): A set of detections to initiate tracks
                from.

        Returns:
            set[Track]: A set of newly initiated tracks.

        """
        tracks = set()
        for detection in detections:
            state_vector = self._detection_to_vector(detection)
            state = GaussianState(
                state_vector=state_vector,
                covar=self.prior_covariance,
                timestamp=detection.timestamp,
            )
            tracks.add(Track([state]))
        return tracks


class ParticleStateInitiator(_SinglePointInitiator):
    """Initialises a new `ParticleState` track from a single detection."""

    def __init__(self, num_particles: int, prior_covariance: np.ndarray):
        """Initialise the particle state initiator.

        Args:
            num_particles (int): The number of particles to generate for each
                new track.
            prior_covariance (np.ndarray): The covariance matrix used to sample
                the initial particle locations.

        """
        self.num_particles = num_particles
        self.prior_covariance = prior_covariance

    def initiate(self, detections: set[Detection]) -> set[Track]:
        """Initiate a new `ParticleState` track for each detection.

        For each detection, this method generates a cloud of particles sampled
        from a Gaussian distribution centered on the initial state estimate.

        Args:
            detections (set[Detection]): A set of detections to initiate
                tracks from.

        Returns:
            set[Track]: A set of newly initiated tracks.

        """
        tracks = set()
        for detection in detections:
            state_vector = self._detection_to_vector(detection)

            # Generate particles by sampling from a Gaussian distribution
            samples = np.random.multivariate_normal(
                mean=state_vector.flatten(),
                cov=self.prior_covariance,
                size=self.num_particles,
            )

            # Assign equal weights to all particles
            weights = np.ones(self.num_particles) / self.num_particles

            # Create a ParticleState directly from the samples and weights
            prior_state = ParticleState(
                state_vector=samples.T, weights=weights, timestamp=detection.timestamp
            )
            tracks.add(Track([prior_state]))

        return tracks


class MultiMeasurementInitiator(Initiator):
    """A logic-based initiator that confirms tracks only after multiple detections.

    This initiator manages a set of unconfirmed tracks. It attempts to update
    these tracks with new detections at each step. A track is only confirmed
    and returned once it has been updated a minimum number of times.
    """

    def __init__(
        self,
        initiator: Initiator,
        deleter: Deleter,
        predictor: Predictor,
        updater: Updater,
        hypothesiser: Hypothesiser,
        min_points: int,
        time_interval: timedelta,
    ):
        """Initialise the multi-measurement initiator.

        Args:
            initiator (Initiator): A simple initiator used to create new
                unconfirmed tracks from unassociated detections.
            deleter (Deleter): A deleter used to prune unconfirmed tracks.
            predictor (Predictor): The predictor used to advance unconfirmed
                tracks.
            updater (Updater): The updater used to update unconfirmed tracks.
            hypothesiser (Hypothesiser): The hypothesiser used to associate
                detections with unconfirmed tracks.
            min_points (int): The minimum number of states a track must have
                before it is considered confirmed.
            time_interval (timedelta): The time interval between updates.

        """
        self.min_points = min_points
        self.predictor = predictor
        self.updater = updater
        self.hypothesiser = hypothesiser
        self.initiator = initiator
        self.deleter = deleter
        self.time_interval = time_interval
        self.unconfirmed_tracks: set[Track] = set()

    def initiate(self, detections: set[Detection]) -> set[Track]:
        """Update unconfirmed tracks and return any that meet the confirmation criteria.

        Args:
            detections (set[Detection]): The current set of detections.

        Returns:
            set[Track]: A set of newly confirmed tracks.

        """
        if not detections:
            return set()

        confirmed_tracks = set()
        to_remove = set()  # Tracks to remove from unconfirmed set

        # Predict forward the unconfirmed tracks
        priors = {
            track: self.predictor.predict(track[-1], self.time_interval)
            for track in self.unconfirmed_tracks
        }

        # Associate detections with unconfirmed tracks
        hypotheses, unassociated_detections = self.hypothesiser.hypothesise(
            list(priors.values()), list(detections)
        )

        # Update unconfirmed tracks
        for track, prior in priors.items():
            posterior = self.updater.update(hypotheses[prior])
            track.append(posterior)

            # If a track is long enough, confirm it
            if len(track) >= self.min_points:
                confirmed_tracks.add(track)
                to_remove.add(track)

        self.unconfirmed_tracks -= to_remove

        # Prune any unconfirmed tracks that are no longer valid
        self.unconfirmed_tracks -= self.deleter.delete_tracks(self.unconfirmed_tracks)

        # Initiate new unconfirmed tracks from unassociated detections
        self.unconfirmed_tracks |= self.initiator.initiate(unassociated_detections)

        return confirmed_tracks

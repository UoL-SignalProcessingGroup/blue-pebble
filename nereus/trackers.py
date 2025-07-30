"""Defines trackers for single and multi-target tracking.

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
© Copyright 2025 Joshua J. Wakefield.
"""  # noqa: E501

from datetime import timedelta

from nereus.association.hypothesisers import Hypothesiser
from nereus.filters.predictors import Predictor
from nereus.filters.updaters import Updater
from nereus.management.deleters import Deleter
from nereus.management.initiators import Initiator
from nereus.models.measurement import MeasurementModel
from nereus.models.transition import TransitionModel
from nereus.types.detections import Detection
from nereus.types.states import Track


class SingleTargetTracker:
    """Single target tracker."""

    def __init__(
        self,
        track: Track,
        transition_model: TransitionModel,
        measurement_model: MeasurementModel,
        predictor: Predictor,
        updater: Updater,
        hypothesiser: Hypothesiser,
        time_interval: timedelta,
    ):
        """Initialise a SingleTargetTracker.

        Args:
            track (Track): The initial track.
            transition_model (TransitionModel): The transition model.
            measurement_model (MeasurementModel): The measurement model.
            predictor (Predictor): The predictor.
            updater (Updater): The updater.
            hypothesiser (Hypothesiser): The hypothesiser.
            time_interval (timedelta): The time interval between updates.

        """
        self.track = track
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.predictor = predictor
        self.updater = updater
        self.hypothesiser = hypothesiser
        self.time_interval = time_interval

    def update(self, detections: set[Detection]) -> None:
        """Perform a tracking update step.

        Args:
            detections (set[Detection]): The detections in the latest scan.

        """
        # --- Step 1: Predict ---
        prior = self.predictor.predict(self.track[-1], self.time_interval)

        # --- Step 2: Update (or just append prediction if no detections) ---
        if not detections:
            # If no detections, the prediction is the best estimate for the new state
            self.track.append(prior)
        else:
            # If detections exist, perform the full association and update cycle
            hypotheses = self.hypothesiser.hypothesise(prior, detections)
            posterior = self.updater.update(hypotheses)
            self.track.append(posterior)

    def get_track(self) -> Track:
        """Return the current track.

        Returns:
            Track: The current track.

        """
        return self.track


class MultiTargetTracker:
    """Multi-target tracker."""

    def __init__(
        self,
        transition_model: TransitionModel,
        measurement_model: MeasurementModel,
        predictor: Predictor,
        updater: Updater,
        hypothesiser: Hypothesiser,
        time_interval: timedelta,
        tracks: set[Track] = None,
        initiator: Initiator = None,
        deleter: Deleter = None,
    ):
        """Initialise a MultiTargetTracker.

        Args:
            transition_model (TransitionModel): The transition model.
            measurement_model (MeasurementModel): The measurement model.
            predictor (Predictor): The predictor.
            updater (Updater): The updater.
            hypothesiser (Hypothesiser): The hypothesiser.
            time_interval (timedelta): The time interval between updates.
            tracks (set[Track], optional): Initial set of tracks. Defaults to None.
            initiator (Initiator, optional): Track initiator. Defaults to None.
            deleter (Deleter, optional): Track deleter. Defaults to None.

        """
        self.all_tracks = tracks if tracks is not None else set()
        self.current_tracks = tracks.copy() if tracks is not None else set()
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.predictor = predictor
        self.updater = updater
        self.hypothesiser = hypothesiser
        self.time_interval = time_interval
        self.initiator = initiator
        self.deleter = deleter

    def update(self, detections: set[Detection]) -> None:
        """Perform a multi-target tracking update step.

        Args:
            detections (set[Detection]): The detections in the latest scan.

        """
        # --- Step 1: Predict ---
        priors = {
            track: self.predictor.predict(track[-1], self.time_interval)
            for track in self.current_tracks
        }

        # --- Step 2: Update (or just append prediction if no detections) ---
        if not detections:
            # If no detections, the prediction becomes the new state
            for track, prior in priors.items():
                track.append(prior)
        else:
            # If detections exist, proceed with association and update
            prior_list = list(priors.values())
            hypotheses, unassociated_detections = self.hypothesiser.hypothesise(
                prior_list, detections
            )

            for track, prior in priors.items():
                posterior = self.updater.update(hypotheses[prior])
                track.append(posterior)

        # --- Step 3: Delete and Initiate (if configured) ---
        if self.initiator and self.deleter:
            # Note: unassociated_detections is only defined if detections exist
            unassoc_dets = unassociated_detections if detections else set()
            self.current_tracks -= self.deleter.delete(self.current_tracks)
            self.current_tracks |= self.initiator.initiate(unassoc_dets)

        self.all_tracks |= self.current_tracks

    def get_tracks(self) -> set[Track]:
        """Return all tracks.

        Returns:
            set[Track]: All tracks.

        """
        return self.all_tracks


class HierarchicalMultiTargetTracker:
    """Hierarchical multi-target tracker (not implemented)."""

    pass

"""Provide classes for generating hypotheses for data association in target tracking.

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

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
from pyehm.core import EHM2
from scipy.spatial.distance import mahalanobis
from scipy.special import gamma
from scipy.stats import chi2, multivariate_normal

from nereus.models.measurement import MeasurementModel
from nereus.types.detections import Detection, MissedDetection
from nereus.types.hypotheses import Hypothesis, ProbabilityHypothesis
from nereus.types.states import GaussianState, ParticleState, State

# A type alias for states handled by this class
StateType: TypeAlias = GaussianState | ParticleState

# Define a constant for the missed detection column index
MISSED_DETECTION_INDEX = 0


class Hypothesiser(ABC):
    """Abstract base class for hypothesis generators."""

    @abstractmethod
    def hypothesise(self, state, detections):
        """Generate hypotheses for a given state and a set of detections.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class BasicHypothesiser(Hypothesiser):
    """Creates a simple hypothesis for each detection and a missed detection."""

    def __init__(self, measurement_model: MeasurementModel):
        """Initialise the BasicHypothesiser.

        Args:
            measurement_model (MeasurementModel): The model used to predict
                measurements from states.

        """
        self.measurement_model = measurement_model

    def hypothesise(
        self, state: State, detections: Iterable[Detection]
    ) -> list[Hypothesis]:
        """Generate hypotheses for a given state and a set of detections.

        For a given state, this creates one hypothesis for associating it with
        each detection, and one for a missed detection.

        Args:
            state (State): The track state to hypothesise on.
            detections (Iterable[Detection]): A collection of detections to
                associate with the state.

        Returns:
            list[Hypothesis]: A list containing the generated hypotheses.

        """
        # Predict the measurement from the state
        measurement_prediction = self.measurement_model.function(
            state, noise=False
        ).astype(np.float64)

        # Create a missed detection hypothesis
        hypotheses = [Hypothesis(state, MissedDetection())]

        # Create a hypothesis for each detection
        for measurement in detections:
            hypotheses.append(Hypothesis(state, measurement, measurement_prediction))

        return hypotheses


class ProbabilisticHypothesiser(Hypothesiser, ABC):
    """An abstract base class for gated, probabilistic hypothesisers.

    This class handles common functionality such as initialisation, measurement
    prediction, and gating volume calculation. It assumes a probabilistic
    framework where hypotheses have associated likelihoods or probabilities.
    """

    def __init__(
        self,
        measurement_model: MeasurementModel,
        detection_probability: float,
        gate_probability: float,
        clutter_spatial_density: float | None = None,
    ):
        """Initialise the hypothesiser.

        Args:
            measurement_model (MeasurementModel): The measurement model.
            detection_probability (float): The probability of detecting a target.
            gate_probability (float): The probability mass to include in the
                validation gate. Used to calculate the Mahalanobis distance
                threshold.
            clutter_spatial_density (float, optional): The spatial density of
                clutter. If None, it is assumed to be uniform within the gate
                volume. Defaults to None.

        """
        self.measurement_model = measurement_model
        self.detection_probability = detection_probability
        self.gate_probability = gate_probability
        self.clutter_spatial_density = clutter_spatial_density

        # Pre-calculate the gating threshold from the chi-squared distribution
        self.gate_threshold = chi2.ppf(
            gate_probability, df=len(self.measurement_model.mapping)
        )

    def _predict_measurement_and_covar(
        self, state: StateType
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the measurement and its covariance for a given state.

        Args:
            state (GaussianState | ParticleState): The state to predict from.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the predicted
            measurement mean and its covariance.

        """
        if isinstance(state, ParticleState):
            # For particles, predict and calculate covariance from the distribution
            measurement_prediction = self.measurement_model.function(
                state, noise=False
            ).astype(np.float64)
            mean_prediction = self.measurement_model.function(
                state.mean, noise=False
            ).astype(np.float64)
            covar = np.atleast_2d(np.cov(measurement_prediction, rowvar=True))
        elif isinstance(state, GaussianState):
            # For Gaussian states, project the state and covariance
            mean_prediction = self.measurement_model.function(
                state, noise=False
            ).astype(np.float64)
            H = self.measurement_model.H
            covar = H @ state.covar @ H.T + self.measurement_model.R
        else:
            raise TypeError("State must be either ParticleState or GaussianState")
        return mean_prediction, covar

    def _validation_region_volume(
        self, measurement_prediction: np.ndarray, covar: np.ndarray
    ) -> float:
        """Calculate the volume of the validation region (the gate).

        The volume is that of an n-dimensional ellipsoid defined by the gating
        threshold and the measurement covariance.

        Args:
            measurement_prediction (np.ndarray): The predicted measurement, used
                to determine dimensionality.
            covar (np.ndarray): The covariance of the predicted measurement.

        Returns:
            float: The volume of the validation gate.

        """
        n = measurement_prediction.ndim
        # Volume of a unit n-hypersphere
        c_n = np.pi ** (n / 2) / gamma(n / 2 + 1)
        # Scale by the gate size and the determinant of the covariance
        return c_n * self.gate_threshold ** (n / 2) * np.sqrt(np.linalg.det(covar))


class PDAHypothesiser(ProbabilisticHypothesiser):
    """Probabilistic Data Association (PDA) Hypothesiser.

    Calculates association probabilities for a single track against a set of
    validated detections.
    """

    def hypothesise(
        self,
        state: StateType,
        detections: Iterable[Detection],
    ) -> list[ProbabilityHypothesis]:
        """Generate weighted hypotheses for a track and detections using PDA.

        This method performs gating to select valid detections, calculates the
        likelihood of each detection being associated with the track, and
        computes the likelihood of a missed detection. These likelihoods are
        then normalised to form association probabilities.

        Args:
            state (StateType): The track state to hypothesise on.
            detections (Iterable[Detection]): The detections to associate with
                the state.

        Returns:
            list[ProbabilityHypothesis]: A list of hypotheses, including a
                missed detection, each with a calculated association
                probability.

        """
        # Predict measurement and covariance from the track state
        pred_mean, pred_covar = self._predict_measurement_and_covar(state)
        inv_covar = np.linalg.pinv(pred_covar)

        # Gate detections: only consider detections within the validation gate
        validated_detections = [
            mahalanobis(pred_mean.flatten(), det.state_vector.flatten(), inv_covar)
            <= self.gate_threshold
            for det in detections
        ]

        # Calculate likelihood of missed detection
        missed_detection_likelihood = (
            1 - self.detection_probability * self.gate_probability
        )

        # Calculate likelihood for each validated detection
        likelihoods = [missed_detection_likelihood]
        num_validated = sum(validated_detections)

        for i, detection in enumerate(detections):
            if validated_detections[i]:
                # Likelihood is the PDF of the measurement residual
                pdf = multivariate_normal.pdf(
                    (detection.state_vector - pred_mean).ravel(), cov=pred_covar
                )
                likelihood = pdf * self.detection_probability

                # Adjust for clutter density
                if self.clutter_spatial_density:
                    likelihood /= self.clutter_spatial_density
                elif num_validated > 0:
                    # If no clutter density, assume uniform clutter in the gate
                    volume = self._validation_region_volume(pred_mean, pred_covar)
                    likelihood *= volume / num_validated
                likelihoods.append(likelihood)
            else:
                # Detections outside the gate have zero likelihood
                likelihoods.append(0)

        # Normalise all likelihoods to get association probabilities
        total_likelihood = sum(likelihoods)
        if total_likelihood > 0:
            probabilities = [lh / total_likelihood for lh in likelihoods]
        else:
            # If no associations, probability of missed detection is 1
            probabilities = [1.0] + [0.0] * len(detections)

        # Create the final ProbabilityHypothesis objects
        hypotheses = [
            ProbabilityHypothesis(
                state, MissedDetection(), probability=probabilities[0]
            )
        ]
        for i, detection in enumerate(detections):
            hypotheses.append(
                ProbabilityHypothesis(state, detection, probabilities[i + 1], pred_mean)
            )

        return hypotheses


class JPDAHypothesiser(ProbabilisticHypothesiser):
    """Joint Probabilistic Data Association (JPDA) Hypothesiser.

    This class generates joint hypotheses for multiple tracks and detections.
    It can use two methods for calculating the association matrix:
    1. A combinatorial approach that evaluates all valid joint hypotheses.
    2. The EHM (Efficient Hypothesis Management) algorithm.
    """

    def __init__(
        self,
        measurement_model: MeasurementModel,
        detection_probability: float,
        gate_probability: float,
        clutter_spatial_density: float | None = None,
        use_ehm: bool = True,
    ):
        """Initialise the JPDA hypothesiser.

        Args:
            measurement_model (MeasurementModel): The measurement model.
            detection_probability (float): Probability of detecting a target.
            gate_probability (float): Probability mass to include in the gate.
            clutter_spatial_density (float, optional): Spatial density of
                clutter. Defaults to None.
            use_ehm (bool, optional): If True, uses the EHM library for
                efficient association. If False, uses a standard combinatorial
                approach. Defaults to True.

        """
        super().__init__(
            measurement_model,
            detection_probability,
            gate_probability,
            clutter_spatial_density,
        )
        self.use_ehm = use_ehm

    def hypothesise(
        self, states: list[StateType], detections: list[Detection]
    ) -> tuple[dict[State, list[ProbabilityHypothesis]], set[Detection]]:
        """Generate joint hypotheses for multiple tracks and detections.

        Args:
            states (list[State]): A list of track states.
            detections (list[Detection]): A list of detections.

        Returns:
            tuple[dict[State, list[ProbabilityHypothesis]], set[Detection]]:
                A tuple containing:
                - A dictionary mapping each track state to its list of
                  hypotheses (including missed detection).
                - A set of detections that were not associated with any track.

        """
        num_states = len(states)
        predictions = [self._predict_measurement_and_covar(state) for state in states]
        pred_means, pred_covars = zip(*predictions)

        validation_matrix, unassociated = self._compute_validation_matrix(
            detections, list(pred_means), list(pred_covars)
        )

        likelihood_matrix = self._compute_likelihood_matrix(
            num_states,
            detections,
            validation_matrix,
            list(pred_means),
            list(pred_covars),
        )

        association_matrix = self._compute_association_matrix(
            validation_matrix, likelihood_matrix
        )

        hypotheses = self._generate_hypotheses(
            states, detections, association_matrix, list(pred_means)
        )

        return hypotheses, unassociated

    def _compute_validation_matrix(
        self,
        detections: list[Detection],
        pred_means: list[np.ndarray],
        pred_covars: list[np.ndarray],
    ) -> tuple[np.ndarray, set[Detection]]:
        """Compute a boolean matrix of valid track-detection pairs.

        This method also identifies detections that fall outside all validation
        gates.

        Args:
            detections (list[Detection]): List of detections.
            pred_means (list[np.ndarray]): List of predicted measurement means.
            pred_covars (list[np.ndarray]): List of predicted measurement
                covariances.

        Returns:
            tuple[np.ndarray, set[Detection]]: A tuple containing:
                - The boolean validation matrix (rows=tracks, cols=detections).
                - A set of unassociated detections.

        """
        num_states = len(pred_means)
        num_detections = len(detections)
        matrix = np.zeros((num_states, num_detections + 1), dtype=bool)
        matrix[:, MISSED_DETECTION_INDEX] = True  # Missed detection is always valid

        if num_detections == 0:
            return matrix, set()

        # Stack detection vectors into a single NumPy array for vectorization
        detection_vectors = np.array([d.state_vector.flatten() for d in detections])
        associated_detections_mask = np.zeros(num_detections, dtype=bool)

        for i in range(num_states):
            inv_cov = np.linalg.pinv(pred_covars[i])
            # Calculate Mahalanobis distance for all detections at once
            diff = detection_vectors - pred_means[i].flatten()
            # Equivalent to np.diag(diff @ inv_cov @ diff.T) but more efficient
            distances = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)

            # Update validation matrix and associated detection mask
            valid_mask = distances <= self.gate_threshold
            matrix[i, 1:] = valid_mask
            associated_detections_mask |= valid_mask

        unassociated = set(itertools.compress(detections, ~associated_detections_mask))
        return matrix, unassociated

    def _compute_likelihood_matrix(
        self,
        num_states: int,
        detections: list[Detection],
        validation_matrix: np.ndarray,
        pred_means: list[np.ndarray],
        pred_covars: list[np.ndarray],
    ) -> np.ndarray:
        """Compute the likelihood for each valid track-detection pair.

        If clutter_spatial_density is not provided, it is calculated
        dynamically for each track based on the number of validated detections
        within its gate.

        Args:
            num_states (int): Number of tracks.
            detections (list[Detection]): List of detections.
            validation_matrix (np.ndarray): Boolean validation matrix.
            pred_means (list[np.ndarray]): List of predicted measurement means.
            pred_covars (list[np.ndarray]): List of predicted measurement
                covariances.

        Returns:
            np.ndarray: The matrix of likelihood values.

        """
        num_detections = len(detections)
        matrix = np.zeros((num_states, num_detections + 1))

        det_prob = self.detection_probability
        if not isinstance(det_prob, (list, np.ndarray)):
            det_prob = [det_prob] * num_states

        detection_vectors = np.array([d.state_vector.flatten() for d in detections])

        for i in range(num_states):
            # Calculate missed detection likelihood
            matrix[i, MISSED_DETECTION_INDEX] = 1 - det_prob[i] * self.gate_probability

            # Get the mask of validated detections for this track
            valid_mask = validation_matrix[i, 1:]
            if not np.any(valid_mask):
                continue

            # Calculate likelihoods for all validated detections at once
            pdf = multivariate_normal.pdf(
                detection_vectors[valid_mask],
                mean=pred_means[i].ravel(),
                cov=pred_covars[i],
            )

            if self.clutter_spatial_density is not None:
                likelihood = (pdf * det_prob[i]) / self.clutter_spatial_density
            else:
                num_validated = valid_mask.sum()
                volume = self._validation_region_volume(pred_means[i], pred_covars[i])
                likelihood = (pdf * det_prob[i] * volume) / num_validated

            matrix[i, 1:][valid_mask] = likelihood

        return matrix

    def _compute_association_matrix(
        self, validation_matrix: np.ndarray, likelihood_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute the association probability matrix.

        This method dispatches to either the EHM library or the combinatorial
        approach based on the `use_ehm` flag.

        Args:
            validation_matrix (np.ndarray): Boolean validation matrix.
            likelihood_matrix (np.ndarray): Likelihood matrix.

        Returns:
            np.ndarray: The association probability matrix.

        """
        if self.use_ehm:
            return EHM2.run(validation_matrix, likelihood_matrix)
        else:
            return self._compute_association_combinatorial(
                validation_matrix, likelihood_matrix
            )

    def _compute_association_combinatorial(
        self, validation_matrix: np.ndarray, likelihood_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate association probabilities using a combinatorial approach.

        Args:
            validation_matrix (np.ndarray): Boolean validation matrix.
            likelihood_matrix (np.ndarray): Likelihood matrix.

        Returns:
            np.ndarray: The association probability matrix.

        """
        num_tracks, _ = validation_matrix.shape

        possible_assoc = [
            list(np.flatnonzero(validation_matrix[t, :])) for t in range(num_tracks)
        ]
        joint_hyps = itertools.product(*possible_assoc)

        def is_valid(hyp):
            detections = [d for d in hyp if d != 0]
            return len(detections) == len(set(detections))

        valid_hyps = [hyp for hyp in joint_hyps if is_valid(hyp)]

        hyp_likelihoods = {
            hyp: np.prod([likelihood_matrix[t, d] for t, d in enumerate(hyp)])
            for hyp in valid_hyps
        }

        total_likelihood = sum(hyp_likelihoods.values())
        if total_likelihood == 0:
            return np.zeros_like(likelihood_matrix)

        matrix = np.zeros_like(likelihood_matrix)
        for t in range(num_tracks):
            for d in range(validation_matrix.shape[1]):
                if validation_matrix[t, d]:
                    marginal_sum = sum(
                        lh for hyp, lh in hyp_likelihoods.items() if hyp[t] == d
                    )
                    matrix[t, d] = marginal_sum

        row_sums = matrix.sum(axis=1, keepdims=True)
        np.divide(matrix, row_sums, out=matrix, where=row_sums != 0)

        return matrix

    def _generate_hypotheses(
        self,
        states: list[State],
        detections: list[Detection],
        association_matrix: np.ndarray,
        pred_means: list[np.ndarray],
    ) -> dict[State, list[ProbabilityHypothesis]]:
        """Construct the final ProbabilityHypothesis objects from the results.

        Args:
            states (list[State]): List of track states.
            detections (list[Detection]): List of detections.
            association_matrix (np.ndarray): Association probability matrix.
            pred_means (list[np.ndarray]): List of predicted measurement means.

        Returns:
            dict[State, list[ProbabilityHypothesis]]: A dictionary mapping each
                state to its list of hypotheses.

        """
        hypotheses = {}
        for i, state in enumerate(states):
            state_hyps = [
                ProbabilityHypothesis(
                    state, MissedDetection(), association_matrix[i, 0]
                )
            ]
            for j, detection in enumerate(detections):
                state_hyps.append(
                    ProbabilityHypothesis(
                        state,
                        detection,
                        association_matrix[i, j + 1],
                        pred_means[i],
                    )
                )
            hypotheses[state] = state_hyps
        return hypotheses

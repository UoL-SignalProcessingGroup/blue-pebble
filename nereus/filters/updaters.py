"""Provide updater implementations for state estimation.

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
from collections.abc import Iterable

import numpy as np
from scipy.stats import multivariate_normal

from nereus.filters.resamplers import Resampler
from nereus.models.measurement import MeasurementModel
from nereus.models.mixture import GaussianMixture
from nereus.types.hypotheses import Hypothesis, ProbabilityHypothesis
from nereus.types.states import GaussianState, ParticleState


class Updater(ABC):
    """Abstract base class for updaters."""

    @abstractmethod
    def update(self, hypothesis):
        """Update a state based on a hypothesis or hypotheses.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class KalmanUpdater(Updater):
    """A standard Kalman filter updater."""

    def __init__(self, measurement_model: MeasurementModel):
        """Initialise the Kalman updater.

        Args:
            measurement_model (MeasurementModel): The measurement model used for
                the update.

        """
        self.measurement_model = measurement_model

    def update(self, hypothesis: Hypothesis) -> GaussianState:
        """Perform the Kalman update step.

        Args:
            hypothesis (Hypothesis): The hypothesis containing the predicted
                state and the measurement to be used for the update.

        Returns:
            GaussianState: The posterior state after the update.

        """
        prediction = hypothesis.prediction
        measurement = hypothesis.measurement
        pred_measurement = hypothesis.measurement_prediction

        # Get measurement model matrix (Jacobian)
        H = self.measurement_model.H

        # Compute innovation and its covariance
        innovation = measurement.state_vector - pred_measurement
        innovation_covar = self.measurement_model.R + H @ prediction.covar @ H.T

        # Compute Kalman gain
        innovation_covar_inv = np.linalg.pinv(innovation_covar)
        kalman_gain = prediction.covar @ H.T @ innovation_covar_inv

        # Update state mean and covariance
        updated_mean = prediction.state_vector + kalman_gain @ innovation
        updated_cov = (
            np.eye(prediction.state_vector.shape[0]) - kalman_gain @ H
        ) @ prediction.covar

        return GaussianState(updated_mean, updated_cov, timestamp=prediction.timestamp)


class GaussianMixtureKalmanUpdater(KalmanUpdater):
    """A Kalman updater to handle multiple hypotheses to produce a Gaussian mixture."""

    def update(self, hypotheses: Iterable[ProbabilityHypothesis]) -> GaussianState:
        """Update a state using a weighted mixture of hypotheses.

        This method performs a Kalman update for each hypothesis, weights the
        resulting posterior state by the hypothesis probability, and then
        reduces the resulting Gaussian mixture to a single Gaussian state.

        Args:
            hypotheses (Iterable[ProbabilityHypothesis]): A collection of
                hypotheses, each with an associated probability.

        Returns:
            GaussianState: The final reduced posterior state.

        """
        posterior_states = []
        posterior_state_weights = []
        for hypothesis in hypotheses:
            # For a missed detection, the posterior is the same as the prior
            if not hypothesis:
                posterior_states.append(hypothesis.prediction)
            else:
                posterior_state = super().update(hypothesis)
                posterior_states.append(posterior_state)
            posterior_state_weights.append(hypothesis.probability)

        # Create a Gaussian mixture from the posterior states
        posterior_mixture = GaussianMixture(posterior_states, posterior_state_weights)

        # Reduce the mixture to a single Gaussian state
        return posterior_mixture.reduce()


class ParticleUpdater(Updater):
    """A standard particle filter updater."""

    def __init__(self, measurement_model: MeasurementModel, resampler: Resampler):
        """Initialise the particle updater.

        Args:
            measurement_model (MeasurementModel): The measurement model.
            resampler (Resampler): The resampler to be used after the update.

        """
        self.measurement_model = measurement_model
        self.resampler = resampler

    def update(self, hypothesis: Hypothesis) -> ParticleState:
        """Perform the particle filter update step (without resampling).

        This method computes new particle weights based on the likelihood of
        the measurement given the particle predictions.

        Args:
            hypothesis (Hypothesis): The hypothesis containing the particle
                state and the measurement.

        Returns:
            ParticleState: The posterior particle state with updated weights.

        """
        predicted_measurements = self.measurement_model.function(
            hypothesis.prediction, noise=False
        )
        measurement = hypothesis.measurement

        # Compute likelihood of the measurement given the predicted measurements
        likelihoods = multivariate_normal.pdf(
            (measurement.state_vector - predicted_measurements).T,
            cov=self.measurement_model.R,
        )

        # Compute new weights for the particles
        new_weights = hypothesis.prediction.weights * likelihoods

        # Normalize weights to sum to 1
        total_weight = np.sum(new_weights)
        if total_weight > 0:
            new_weights /= total_weight

        # Create a new ParticleState with the original particles but updated weights
        return ParticleState(
            state_vector=hypothesis.prediction.state_vector,
            weights=new_weights,
            timestamp=hypothesis.prediction.timestamp,
        )


class ExpectedLikelihoodParticleUpdater(ParticleUpdater):
    """A particle updater that uses the expected likelihood for multiple hypotheses."""

    def update(self, hypotheses: Iterable[ProbabilityHypothesis]) -> ParticleState:
        """Update a particle state using a weighted mixture of hypotheses.

        This method calculates the posterior particle weights as the weighted
        sum of the likelihoods over all hypotheses. This combined state is then
        resampled once to generate the final posterior particle state.

        Args:
            hypotheses (Iterable[ProbabilityHypothesis]): A collection of
                hypotheses, each with an associated probability.

        Returns:
            ParticleState: The final resampled posterior particle state.

        """
        # Store weight updates for each hypothesis
        particle_weights_per_hypothesis = []
        for hypothesis in hypotheses:
            if not hypothesis:  # Missed detection case
                # Likelihood is 1, so weight is probability * prior weight
                particle_weights_per_hypothesis.append(
                    hypothesis.probability * hypothesis.prediction.weights
                )
            else:
                # Compute updated particle weights for this hypothesis
                updated_state = super().update(hypothesis)
                particle_weights_per_hypothesis.append(
                    updated_state.weights * hypothesis.probability
                )

        # Sum the weighted contributions from all hypotheses
        new_weights = np.sum(particle_weights_per_hypothesis, axis=0)

        # Normalize weights to sum to 1
        total_weight = np.sum(new_weights)
        if total_weight > 0:
            new_weights /= total_weight

        # Create a new ParticleState with the original particles and the final summed
        # weights. The state_vector is taken from the first hypothesis's prediction, as
        # all hypotheses share the same prior state.
        new_state = ParticleState(
            state_vector=hypotheses[0].prediction.state_vector,
            weights=new_weights,
            timestamp=hypotheses[0].prediction.timestamp,
        )

        # Perform resampling to mitigate particle degeneracy
        return self.resampler.resample(new_state)

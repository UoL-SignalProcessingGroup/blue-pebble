"""Defines mixture model classes for state representation and reduction.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np

from nereus.types.arrays import StateVector
from nereus.types.states import GaussianState, State


class MixtureModel(ABC):
    """Abstract base class for mixture model representations."""

    @abstractmethod
    def reduce(self) -> State:
        """Reduce the mixture to a single state representation.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class GaussianMixture(MixtureModel):
    """A Gaussian Mixture representation of a state."""

    def __init__(self, components: list[GaussianState], weights: list[float]):
        """Initialise a Gaussian Mixture.

        Args:
            components (list[GaussianState]): A list of Gaussian states
                representing the components of the mixture.
            weights (list[float]): A list of weights corresponding to each
                component.

        """
        if not components:
            raise ValueError("Component list cannot be empty.")
        self.components = components
        self.weights = np.asarray(weights) / np.sum(weights)
        self.timestamp = components[0].timestamp

    def reduce(self) -> GaussianState:
        """Reduce the Gaussian Mixture to a single equivalent Gaussian State.

        This is done by computing the weighted mean and covariance of the
        mixture components.

        Returns:
            GaussianState: A single Gaussian state representing the mixture.

        """
        mean = self._mean()
        covar = self._covar(mean)
        return GaussianState(mean, covar, timestamp=self.timestamp)

    def _mean(self) -> StateVector:
        """Compute the weighted mean of the components.

        Returns:
            StateVector: The weighted mean of the components.

        """
        means = np.hstack([component.state_vector for component in self.components])
        mean = np.average(means, axis=1, weights=self.weights)
        return StateVector(mean)

    def _covar(self, mean: StateVector) -> np.ndarray:
        """Compute the weighted covariance of the components.

        This implements the formula for the covariance of a mixture of
        Gaussians, which includes the weighted sum of component covariances
        and the spread of the component means.

        Args:
            mean (StateVector): The weighted mean of the mixture.

        Returns:
            np.ndarray: The weighted covariance of the components.

        """
        # First term: Weighted sum of the covariances of each component
        covars = np.stack([component.covar for component in self.components], axis=2)
        weighted_covars = np.sum(covars * self.weights, axis=2)

        # Second term: The spread of the means around the overall mean
        means = np.hstack([component.state_vector for component in self.components])
        delta_means = means - mean
        # This calculates Sum(w_i * (d_i)(d_i)^T)
        spread_of_means = np.einsum(
            "i,ji,ki->jk", self.weights, delta_means, delta_means
        )

        return weighted_covars + spread_of_means

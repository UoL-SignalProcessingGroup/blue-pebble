"""Define transition models for tracking systems.

This module is a derivative work of Stone Soup, available at
https://github.com/dstl/StoneSoup.

The original work is licensed under the MIT License.
© Crown Copyright 2017-2025 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2025 Defence Research and Development Canada / Recherche et développement pour la défense Canada  # noqa: E501
© Copyright 2018-2025 University of Liverpool UK
© Copyright 2020-2025 Fraunhofer FKIE
© Copyright 2020-2025 John Hiles
© Copyright 2020-2025 Riskaware Ltd
© Copyright 2021-2025 Roke Manor Research Ltd UK
© Copyright 2023-2025 Loughborough University UK
© Copyright 2025 Joshua J. Wakefield.
"""  # noqa: E501

from abc import ABC, abstractmethod
from datetime import timedelta

import numpy as np
from scipy.linalg import block_diag

from nereus.types.states import State


class TransitionModel(ABC):
    """Abstract base class for state transition prediction models."""

    @abstractmethod
    def function(self, state: State, noise: bool = False) -> np.ndarray:
        """Predict the next state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """The process noise covariance matrix Q."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ndim_state(self) -> int:
        """The number of dimensions in the state vector."""
        raise NotImplementedError


class LinearGaussianTransitionModel(TransitionModel):
    """Parent class for linear Gaussian state transition models.

    This class provides the common prediction logic and properties for models
    defined by a state transition matrix (F) and process noise matrix (Q).
    """

    @property
    def ndim_state(self) -> int:
        """The number of dimensions in the state vector."""
        return self._ndim_state

    @property
    def Q(self) -> np.ndarray:
        """The process noise covariance matrix Q."""
        return self._Q

    def function(self, state: State, noise: bool = False) -> np.ndarray:
        """Predict the next state by applying the transition model."""
        next_state = self.F @ state.state_vector
        if noise:
            noise_vector = np.random.multivariate_normal(
                mean=np.zeros(self.ndim_state),
                cov=self.Q,
                size=state.state_vector.shape[1],
            ).T
            next_state = next_state + noise_vector
        return next_state


class CombinedLinearGaussianTransitionModel(TransitionModel):
    """Combines multiple transition models into a single prediction model."""

    def __init__(self, models: list[TransitionModel]):
        """Initialise the combined model from a list of sub-models.

        Args:
            models (List[TransitionModel]): A list of configured transition
                model instances (e.g., [NCV(), NCV()]) for a 2D system.

        """
        # Combine the F and Q matrices from the sub-models
        self.F: np.ndarray = block_diag(*[model.F for model in models])
        self._Q: np.ndarray = block_diag(*[model.Q for model in models])

    @property
    def ndim_state(self) -> int:
        """The number of dimensions in the state vector."""
        return self.F.shape[0]

    @property
    def Q(self) -> np.ndarray:
        """The process noise covariance matrix Q."""
        return self._Q

    def function(self, state: State, noise: bool = False) -> np.ndarray:
        """Predict the next state by applying the combined transition model."""
        # The logic is identical to the base classes, but uses the combined matrices
        next_state = self.F @ state.state_vector
        if noise:
            noise_vector = np.random.multivariate_normal(
                mean=np.zeros(self.ndim_state),
                cov=self.Q,
                size=state.state_vector.shape[1],
            ).T
            next_state = next_state + noise_vector
        return next_state


class NearlyConstantVelocity(LinearGaussianTransitionModel):
    """A 1D Nearly Constant Velocity model."""

    def __init__(self, noise_std: float, time_interval: timedelta):
        """Initialise the Constant Velocity model for a fixed time interval.

        Args:
            noise_std (float): The standard deviation of the un-modeled
                random accelerations.
            time_interval (timedelta): The fixed time step for the transition.

        """
        dt = time_interval.total_seconds()
        self._ndim_state = 2
        self.F = np.array([[1, dt], [0, 1]])
        self._Q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * noise_std**2


class NearlyConstantAcceleration(LinearGaussianTransitionModel):
    """A 1D Nearly Constant Acceleration model."""

    def __init__(self, noise_std: float, time_interval: timedelta):
        """Initialise the Constant Acceleration model for a fixed time interval.

        Args:
            noise_std (float): The standard deviation of the un-modeled
                random jerk.
            time_interval (timedelta): The fixed time step for the transition.

        """
        dt = time_interval.total_seconds()
        self._ndim_state = 3
        self.F = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
        self._Q = (
            np.array(
                [
                    [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                    [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                    [dt**3 / 6, dt**2 / 2, dt],
                ]
            )
            * noise_std**2
        )

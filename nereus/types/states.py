"""Defines the state classes to represent different types of states in a tracking system.

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

from collections.abc import MutableSequence
from datetime import datetime
from typing import Any, overload

import numpy as np

from nereus.types.arrays import CovarianceMatrix, StateVector, StateVectors


class State:
    """A basic state implementation with a state vector and timestamp."""

    def __init__(
        self,
        state_vector: np.ndarray | StateVector | StateVectors,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialise a BasicState.

        Args:
            state_vector (np.ndarray | StateVector | StateVectors): The state vector.
                If a raw numpy array is provided, it will be converted to a
                `StateVector` (1D) or `StateVectors` (2D) object.
            timestamp (datetime, optional): The timestamp of the state.
                Defaults to None.

        """
        if isinstance(state_vector, np.ndarray) and not isinstance(
            state_vector, (StateVector, StateVectors)
        ):
            if state_vector.ndim == 1:
                state_vector = StateVector(state_vector)
            elif state_vector.ndim == 2:
                state_vector = StateVectors(state_vector)
            else:
                raise ValueError("state_vector must be a 1D or 2D array.")

        self._state_vector = state_vector
        self._timestamp = timestamp

    @property
    def state_vector(self) -> StateVector | StateVectors:
        """The state vector of the state."""
        return self._state_vector

    @property
    def timestamp(self) -> datetime | None:
        """The timestamp of the state."""
        return self._timestamp


class GroundTruthState(State):
    """A state representing the ground truth of a target."""


class GaussianState(State):
    """A state represented by a Gaussian distribution (mean and covariance).

    Attributes:
        mean (StateVector): The mean of the Gaussian state. This is an alias
            for `state_vector`.
        covar (CovarianceMatrix): The covariance matrix of the state.

    """

    def __init__(
        self,
        mean: StateVector | np.ndarray,
        covar: CovarianceMatrix | np.ndarray,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialise a GaussianState.

        Args:
            mean (StateVector | np.ndarray): The mean vector of the Gaussian state.
                If a raw numpy array is provided, it will be converted to a
                `StateVector` object.
            covar (CovarianceMatrix | np.ndarray): The covariance matrix of the state.
                If a raw numpy array is provided, it will be converted to a
                `CovarianceMatrix` object.
            timestamp (datetime, optional): The timestamp of the state.
                Defaults to None.

        """
        super().__init__(mean, timestamp)
        if isinstance(covar, np.ndarray) and not isinstance(covar, CovarianceMatrix):
            covar = CovarianceMatrix(covar)
        self.covar = covar

    @property
    def mean(self) -> StateVector:
        """The mean of the Gaussian state."""
        return self.state_vector


class ParticleState(State):
    """A state represented by a set of weighted particles.

    The `state_vector` for this class is the entire array of particles.
    A single point estimate can be accessed via the `mean` property.

    Attributes:
        state_vector (np.ndarray): An array of particles representing the state.
            Shape is (num_dimensions, num_particles).
        weights (np.ndarray): An array of weights for each particle.
            Shape is (num_particles,).
        timestamp (datetime | None): The timestamp of the state. Defaults to None.

    """

    def __init__(
        self,
        state_vector: StateVectors | np.ndarray,
        weights: np.ndarray,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialise a ParticleState.

        Args:
            state_vector (StateVectors | np.ndarray): An array of particles representing
                the state. Shape is (num_dimensions, num_particles).
            weights (np.ndarray): An array of weights for each particle.
                Shape is (num_particles,).
            timestamp (datetime | None, optional): The timestamp of the state.
                Defaults to None.

        """
        self._state_vector = state_vector
        self.weights = weights
        self._timestamp = timestamp

    @property
    def mean(self) -> StateVector:
        """The weighted mean of the particle state."""
        return StateVector(np.average(self.state_vector, axis=1, weights=self.weights))

    @property
    def covar(self) -> CovarianceMatrix:
        """The weighted covariance of the particle state."""
        return CovarianceMatrix(np.cov(self.state_vector, aweights=self.weights))

    @property
    def num_particles(self) -> int:
        """The number of particles in the state."""
        return self.state_vector.shape[1]


class _Path(MutableSequence):
    """An abstract base class for a sequence of states.

    Provides list-like functionality.
    """

    def __init__(
        self, states: list[State] | None = None, id: Any | None = None
    ) -> None:
        self.states = states if states is not None else []
        self.id = id

    @overload
    def __getitem__(self, index: int) -> State: ...
    @overload
    def __getitem__(self, index: slice) -> list[State]: ...
    def __getitem__(self, index):
        return self.states[index]

    def __setitem__(self, index: int, value: State) -> None:
        self.states[index] = value

    def __delitem__(self, index: int) -> None:
        del self.states[index]

    def __len__(self) -> int:
        return len(self.states)

    def insert(self, index: int, value: State) -> None:
        self.states.insert(index, value)


class Track(_Path):
    """A class to represent the estimated track of a target.

    A track is a sequence of `GaussianState` objects that represents the
    estimated trajectory of an object over time. It behaves like a list.
    """


class GroundTruthPath(_Path):
    """A class to represent the ground truth path of a target.

    A ground truth path is a sequence of `GroundTruthState` objects. It
    provides special indexing to retrieve a state by its exact `datetime`.
    """

    @overload
    def __getitem__(self, index: int) -> GroundTruthState: ...
    @overload
    def __getitem__(self, index: slice) -> list[GroundTruthState]: ...
    @overload
    def __getitem__(self, index: datetime) -> GroundTruthState: ...
    def __getitem__(self, index):
        """Get a state by index or timestamp."""
        if isinstance(index, datetime):
            for state in self.states:
                if state.timestamp == index:
                    return state
            raise IndexError("Timestamp not found in states.")
        return super().__getitem__(index)

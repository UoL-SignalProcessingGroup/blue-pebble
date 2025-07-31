"""Provide measurement model implementations for state estimation.

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

import numpy as np

from nereus.types.angles import Bearing
from nereus.types.states import State


class MeasurementModel(ABC):
    """Abstract base class for measurement models."""

    def __init__(
        self,
        ndim_state: int,
        mapping: tuple[int, ...],
        noise_covar: np.ndarray,
        translation_offset: np.ndarray | None = None,
    ):
        """Initialise the Measurement Model."""
        self.ndim_state = ndim_state
        self.mapping = mapping
        self._R = noise_covar
        self.translation_offset = (
            translation_offset
            if translation_offset is not None
            else np.zeros((len(mapping), 1))
        )

    @property
    def ndim_meas(self) -> int:
        """The number of dimensions in the measurement vector."""
        return self._R.shape[0]

    @property
    def R(self) -> np.ndarray:
        """The measurement noise covariance matrix R."""
        return self._R

    @abstractmethod
    def function(self, state: State, noise: bool = False) -> np.ndarray:
        """Compute the measurement from a given state.

        Args:
            state (State): The state of the system.
            noise (bool, optional): If True, add measurement noise to the
                output. Defaults to False.

        Returns:
            np.ndarray: The corresponding measurement vector.

        """
        raise NotImplementedError

    @abstractmethod
    def inverse_function(self, state: State) -> np.ndarray:
        """Compute the state vector from a given measurement state.

        Args:
            state (State): The measurement state.

        Returns:
            np.ndarray: The corresponding state vector.

        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, state: State) -> np.ndarray:
        """Return the Jacobian matrix of the measurement function.

        For linear models, this is the H matrix. For non-linear models, it is
        the matrix of first-order partial derivatives.

        Args:
            state (State): The state at which to compute the Jacobian.

        Returns:
            np.ndarray: The Jacobian matrix.

        """
        raise NotImplementedError

    def _get_state_vector_from_input(self, state: State | np.ndarray) -> np.ndarray:
        """Extract the mapped state vector from a State object or NumPy array."""
        if isinstance(state, State):
            return state.state_vector[self.mapping, :]
        return state[self.mapping, :]


class LinearGaussianMeasurementModel(MeasurementModel):
    """A linear model mapping a system state to a measurement."""

    @property
    def H(self) -> np.ndarray:
        """The measurement matrix H."""
        ndim_mapped = len(self.mapping)
        H = np.zeros((self.ndim_meas, ndim_mapped))
        H[np.arange(self.ndim_meas), np.arange(self.ndim_meas)] = 1
        return H

    def jacobian(self, state: State | None = None) -> np.ndarray:
        """Return the Jacobian matrix (H) for the linear model."""
        H = np.zeros((self.ndim_meas, self.ndim_state))
        H[np.arange(self.ndim_meas), self.mapping] = 1
        return H

    def function(self, state: State | np.ndarray, noise: bool = False) -> np.ndarray:
        """Compute the measurement using the linear transformation y = Hx - c.

        Args:
            state (State | np.ndarray): The state of the system.
            noise (bool, optional): If True, add measurement noise. Defaults to False.

        Returns:
            np.ndarray: The measurement vector.

        """
        # 1. First, select the relevant components from the state vector
        mapped_state_vector = self._get_state_vector_from_input(state)

        # 2. Then, apply the linear transformation H
        measurement = self.H @ mapped_state_vector - self.translation_offset

        # 3. Finally, add noise if requested
        if noise:
            noise_sample = np.random.multivariate_normal(
                np.zeros(self.ndim_meas), self.R, size=measurement.shape[1]
            ).T
            measurement = measurement + noise_sample

        return measurement

    def inverse_function(self, state: State | np.ndarray) -> np.ndarray:
        """Compute the full state vector from a given measurement state."""
        # Extract the numerical vector from the measurement state
        if isinstance(state, State):
            measurement_vector = state.state_vector
        else:
            measurement_vector = state

        # Invert the measurement to get the value of the mapped state components
        mapped_state = np.linalg.pinv(self.H) @ (
            measurement_vector + self.translation_offset
        )

        # Create a full state vector and place the result in the correct positions
        full_state = np.zeros((self.ndim_state, mapped_state.shape[1]))
        full_state[self.mapping, :] = mapped_state

        return full_state


class CartesianToRangeBearingMeasurementModel(MeasurementModel):
    """A non-linear model to convert 2D Cartesian coordinates to range and bearing.

    This model assumes that the state vector contains Cartesian position
    coordinates (x, y) at the indices specified by the `mapping` attribute.
    """

    def function(self, state: State | np.ndarray, noise: bool = False) -> np.ndarray:
        """Compute the range and bearing from a Cartesian state vector.

        Args:
            state (State | np.ndarray): The state of the system.
            noise (bool, optional): If True, add measurement noise. Defaults to False.

        Returns:
            np.ndarray: The range and bearing measurement vector.

        """
        state_vector = self._get_state_vector_from_input(state)
        x, y = state_vector[0, :], state_vector[1, :]

        rho = np.sqrt(x**2 + y**2)
        phi = np.array([Bearing(i) for i in np.arctan2(y, x)])
        measurement = np.vstack((rho, phi))

        measurement -= self.translation_offset

        if noise:
            measurement = (
                measurement
                + np.random.multivariate_normal(
                    np.zeros(self.ndim_meas), self.R, size=measurement.shape[1]
                ).T
            )
        return measurement

    def inverse_function(self, state: State) -> np.ndarray:
        """Compute the Cartesian coordinates from a range/bearing state vector.

        Args:
            state (State): The state containing the range and bearing measurement.

        Returns:
            np.ndarray: The Cartesian coordinates corresponding to range and bearing.

        """
        measurement_vector = state.state_vector + self.translation_offset
        rho, phi = measurement_vector[0, 0], measurement_vector[1, 0]

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        return np.array([[x], [y]])

    def jacobian(self, state: State) -> np.ndarray:
        """Compute the Jacobian matrix for the Cartesian to range/bearing conversion."""
        state_vector = self._get_state_vector_from_input(state)
        x, y = state_vector[0], state_vector[1]

        rho = np.sqrt(x**2 + y**2)

        # Handle the case where the state is at the origin
        if np.isclose(rho, 0.0):
            # Return a zero Jacobian as a safe default
            return np.zeros((2, self.ndim_state))

        jac = np.zeros((2, self.ndim_state))
        jac[0, self.mapping[0]] = x / rho
        jac[0, self.mapping[1]] = y / rho
        jac[1, self.mapping[0]] = -y / rho**2
        jac[1, self.mapping[1]] = x / rho**2

        return jac


class CartesianToBearingMeasurementModel(MeasurementModel):
    """A non-linear model that converts 2D Cartesian coordinates to bearing only."""

    def function(self, state: State | np.ndarray, noise: bool = False) -> np.ndarray:
        """Compute the bearing from a Cartesian state vector.

        Args:
            state (State | np.ndarray): The state of the system.
            noise (bool, optional): If True, add measurement noise. Defaults to False.

        Returns:
            np.ndarray: The bearing measurement vector.

        """
        state_vector = self._get_state_vector_from_input(state)
        x, y = state_vector[0, :], state_vector[1, :]

        phi = np.array([Bearing(i) for i in np.arctan2(y, x)]).reshape(1, -1)
        phi -= self.translation_offset

        if noise:
            phi += np.random.multivariate_normal(
                np.zeros(self.ndim_meas), self.R, size=phi.shape[1]
            ).T
        return phi

    def inverse_function(self, state: State) -> np.ndarray:
        """Compute Cartesian coordinates from a bearing measurement.

        This inverse is non-unique, as range is unknown. This implementation
        returns a unit vector in the direction of the bearing.

        Args:
            state (State): The state containing the bearing measurement.

        Returns:
            np.ndarray: The Cartesian coordinates (as a unit vector).

        """
        measurement_vector = state.state_vector + self.translation_offset
        phi = measurement_vector[0, :]
        x = np.cos(phi)
        y = np.sin(phi)
        return np.vstack((x, y))

    def jacobian(self, state: State) -> np.ndarray:
        """Compute the Jacobian matrix for the Cartesian to bearing conversion."""
        state_vector = self._get_state_vector_from_input(state)
        x, y = state_vector[0], state_vector[1]

        rho_sq = x**2 + y**2

        # Handle the case where the state is at the origin
        if np.isclose(rho_sq, 0.0):
            # Return a zero Jacobian as a safe default
            return np.zeros((self.ndim_meas, self.ndim_state))

        jac = np.zeros((1, self.ndim_state))
        jac[0, self.mapping[0]] = -y / rho_sq
        jac[0, self.mapping[1]] = x / rho_sq
        return jac

"""Defines sensor platform classes for representing sensor arrays with imperfections.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np

from nereus.types.arrays import StateVector, StateVectors


class SensorPlatform:
    """A base class to represent a sensor platform."""

    def __init__(self, state_vector, timestamp):
        """Initialise the sensor platform with a state vector and timestamp."""
        self.state_vector = state_vector
        self.timestamp = timestamp


class HorizontalLineArray(SensorPlatform):
    """Represents a linear array of sensors, including physical imperfections.

    This class takes the ideal (error-free) state of a sensor array and
    simulates common real-world errors upon creation.

    Attributes:
        state_vector (np.ndarray): The (M, N) array containing the M-D
            *actual* state of the N sensors after simulating imperfections.
        timestamp (float): The timestamp of the state vector.
        num_sensors (int): The number of sensors in the array.
        spacing (float): The nominal spacing between sensors based on ideal
            positions.
        origin (StateVector): The actual state of the sensor at the origin.
        complex_imperfections (np.ndarray): The combined complex gain/phase
            error term for each sensor. If no errors, this is an array of ones.
        has_gain_phase_errors (bool): A flag that is True only if there are
            non-zero gain or phase errors.

    """

    def __init__(
        self,
        state_vector: np.ndarray,
        timestamp: float,
        position_error_std: float = 0.0,
        gain_error_std: float = 0.0,
        phase_error_std_rad: float = 0.0,
    ):
        """Initialise the sensor array and generate its imperfections."""
        self._ideal_state_vector = state_vector
        self.position_error_std = position_error_std
        self.gain_error_std = gain_error_std
        self.phase_error_std_rad = phase_error_std_rad

        self.num_sensors = self._ideal_state_vector.shape[1]
        self.spacing = np.linalg.norm(
            self._ideal_state_vector[:, 1] - self._ideal_state_vector[:, 0]
        )
        self.has_gain_phase_errors = (
            self.gain_error_std > 0.0 or self.phase_error_std_rad > 0.0
        )

        super().__init__(state_vector=None, timestamp=timestamp)
        self._generate_imperfections()

    def _generate_imperfections(self):
        """Create the actual state_vector and complex imperfections."""
        # --- Generate Position Errors ---
        if self.position_error_std > 0.0:
            position_errors = np.random.normal(
                scale=self.position_error_std, size=self._ideal_state_vector.shape
            )
            self.state_vector = StateVectors(
                [
                    StateVector(self._ideal_state_vector[:, i] + position_errors[:, i])
                    for i in range(self.num_sensors)
                ]
            )
        else:
            self.state_vector = self._ideal_state_vector.copy()

        self.origin = StateVector(self.state_vector[:, 0])

        # --- Generate Complex Gain/Phase Errors ---
        if not self.has_gain_phase_errors:
            self.complex_imperfections = np.ones(self.num_sensors, dtype=complex)
            return

        gain = (
            np.random.normal(loc=1.0, scale=self.gain_error_std, size=self.num_sensors)
            if self.gain_error_std > 0.0
            else np.ones(self.num_sensors)
        )

        phase_rad = (
            np.random.normal(
                loc=0.0, scale=self.phase_error_std_rad, size=self.num_sensors
            )
            if self.phase_error_std_rad > 0.0
            else np.zeros(self.num_sensors)
        )

        self.complex_imperfections = gain * np.exp(1j * phase_rad)

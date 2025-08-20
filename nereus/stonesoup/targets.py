"""Defines acoustic targets for use in Stonesoup.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from stonesoup.movable import MovingMovable
from stonesoup.types.array import StateVector


class AcousticTarget:
    """A target that combines a kinematic state (movable) with an acoustic signature."""

    def __init__(
        self,
        movable: MovingMovable,
        amplitudes_upa: list[float],
        frequencies_hz: list[float],
        phases_rad: list[float],
    ) -> None:
        """Initialise the AcousticTarget.

        Args:
            movable (MovingMovable): An initialised movable object for kinematics.
            amplitudes_upa (list[float]): A list of amplitudes for the tonal signature.
            frequencies_hz (list[float]): A list of frequencies (Hz) for the tonal
                signature.
            phases_rad (list[float]): A list of phase offsets (radians) for the tonal
                signature.

        """
        self.movable = movable
        self.amplitudes_upa = amplitudes_upa
        self.frequencies_hz = frequencies_hz
        self.phases_rad = phases_rad

    def move(self, timestamp: float) -> None:
        """Move the target to a new timestamp.

        This method delegates the movement logic to the internal movable object.

        Args:
            timestamp (float): The timestamp to move the target to.

        """
        self.movable.move(timestamp)

    def get_state(self, timestamp: float) -> StateVector:
        """Get the state of the target at a specific timestamp.

        Args:
            timestamp (float): The timestamp for which to retrieve the state.

        Returns:
            StateVector: The state vector of the target at the specified timestamp.

        """
        return self.movable.get_state(timestamp)

    @property
    def position(self) -> StateVector:
        """The current position of the target."""
        return self.movable.position

    @property
    def states(self) -> list[StateVector]:
        """The history of states of the target."""
        return self.movable.states

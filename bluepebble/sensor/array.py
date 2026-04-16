"""Linear hydrophone array model."""

from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.state import State

from .hydrophone import Hydrophone

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.float64]


class LinearHydrophoneArray(Base):
    """A linear array of hydrophone elements with a dynamic position history.

    The array models a uniformly spaced, straight-line arrangement of
    hydrophone elements.  Positions are computed and stored by calling
    :meth:`move` once per simulation timestep.  The caller (typically
    :class:`~bluepebble.platform.TowedArrayPlatform`) supplies the tow-point
    geometry; the array handles element placement.

    Parameters
    ----------
    elements : list of Hydrophone
        Ordered list of hydrophone elements.  Each must be a distinct
        ``Hydrophone`` instance — do not share instances across positions.
    element_spacing_m : float
        Centre-to-centre spacing between adjacent elements in metres.
    reference_element_idx : int, optional
        Index of the element used as the array phase reference.
        Defaults to ``0``.

    """

    elements: list[Hydrophone] = Property(doc="Ordered list of hydrophone elements.")
    element_spacing_m: float = Property(
        doc="Centre-to-centre spacing between adjacent elements in metres."
    )
    reference_element_idx: int = Property(
        default=0,
        doc="Index of the element used as the array phase reference.",
    )

    @property
    def num_elements(self) -> int:
        """Return the number of elements in the array.

        Returns
        -------
        int
            ``len(self.elements)``.

        """
        return len(self.elements)

    @property
    def state(self) -> State | None:
        """Return the current state of the reference element.

        Returns
        -------
        State or None
            The most recent :class:`~stonesoup.types.state.State` for the
            reference element, or ``None`` before the first :meth:`move` call.

        """
        return self.elements[self.reference_element_idx].state

    def move(
        self,
        leader_pos: ArrayLike,
        trail_direction_xy: ArrayLike,
        cable_length_m: float,
        array_depth_m: float,
        timestamp: datetime,
    ) -> None:
        """Compute element positions and append a timestamped state to each element.

        The first element is placed at the end of the tow cable (accounting for
        the depth difference between the tow point and the array depth).
        Subsequent elements are spaced uniformly along ``trail_direction_xy`` at
        ``array_depth_m``.

        Parameters
        ----------
        leader_pos : ArrayLike
            3-D position ``[x, y, z]`` of the tow point (host vehicle) in metres.
        trail_direction_xy : ArrayLike
            Unit vector ``[dx, dy]`` pointing in the direction the array trails
            behind the host (opposite to heading).
        cable_length_m : float
            Slant distance in metres from the tow point to the first element.
        array_depth_m : float
            Fixed depth in metres at which all elements are deployed.
        timestamp : datetime
            Timestamp associated with this position snapshot.

        Raises
        ------
        ValueError
            If ``cable_length_m`` is shorter than the vertical separation between
            the tow point and ``array_depth_m``.

        """
        lp = np.asarray(leader_pos, dtype=float).flatten()
        td = np.asarray(trail_direction_xy, dtype=float).flatten()

        depth_diff = abs(float(lp[2]) - array_depth_m)
        if cable_length_m**2 < depth_diff**2:
            msg = (
                f"Cable length ({cable_length_m}m) is too short for the depth "
                f"difference ({depth_diff:.3f}m) between tow point and array depth."
            )
            raise ValueError(msg)

        horizontal_cable = float(np.sqrt(cable_length_m**2 - depth_diff**2))
        first_xy = lp[:2] + horizontal_cable * td

        for i, element in enumerate(self.elements):
            element_xy = first_xy + i * self.element_spacing_m * td
            pos = StateVector([float(element_xy[0]), float(element_xy[1]), array_depth_m])
            element.states.append(State(state_vector=pos, timestamp=timestamp))

    def element_states_at(self, timestamp: datetime) -> list[State]:
        """Return the position state of every element at the given timestamp.

        Parameters
        ----------
        timestamp : datetime
            Timestamp to look up in each element's state history.

        Returns
        -------
        list of State
            One :class:`~stonesoup.types.state.State` per element, in element order.

        Raises
        ------
        ValueError
            If any element has no state recorded at ``timestamp``.

        """
        result: list[State] = []
        for i, element in enumerate(self.elements):
            found: State | None = None
            for s in element.states:
                if s.timestamp == timestamp:
                    found = s
                    break
            if found is None:
                msg = (
                    f"No state found for element {i} at timestamp {timestamp!r}. "
                    "Ensure platform.move() has been called for this timestamp."
                )
                raise ValueError(msg)
            result.append(found)
        return result

    def position_matrix_at(self, timestamp: datetime) -> StateVectors:
        """Return element positions as a ``(3, num_elements)`` matrix.

        Parameters
        ----------
        timestamp : datetime
            Timestamp to look up in each element's state history.

        Returns
        -------
        StateVectors
            Array of shape ``(3, num_elements)`` where each column is the
            3-D position of one element.

        Raises
        ------
        ValueError
            If any element has no state at ``timestamp`` (forwarded from
            :meth:`element_states_at`).

        """
        states = self.element_states_at(timestamp)
        return StateVectors(np.hstack([s.state_vector for s in states]))  # type: ignore[return-value]

    def transfer_functions(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Evaluate the transfer function of every element at the given frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate each element's
            :meth:`~HydrophoneResponse.transfer_function`.

        Returns
        -------
        ComplexArray
            Complex array of shape ``(num_elements, num_frequencies)``.

        """
        return np.stack(
            [element.response.transfer_function(frequencies_hz) for element in self.elements],
            axis=0,
        ).astype(np.complex128)

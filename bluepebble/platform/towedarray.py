"""Defines a towed array platform."""

from collections.abc import Sequence
from datetime import datetime
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.platform.base import MultiTransitionMovingPlatform

from ..sensor.array import LinearHydrophoneArray

FloatArray: TypeAlias = NDArray[np.float64]


class TowedArrayPlatform(MultiTransitionMovingPlatform):
    """A Stone Soup compliant platform that tows a linear hydrophone array.

    The platform models a host vehicle towing a :class:`LinearHydrophoneArray`.
    On each call to :meth:`move`, the host state is advanced and the array
    element positions are recomputed from the current heading and depth
    configuration.

    Parameters
    ----------
    sensor_array : LinearHydrophoneArray
        The hydrophone array towed behind the host.
    cable_length_m : float
        Slant distance in metres from the host to the first array element.
    array_depth_m : float
        Depth in metres at which the array is deployed.
    velocity_mapping : Sequence of int or None, optional
        Indices for velocity components in the state vector.  Defaults to
        ``position_mapping + 1`` when ``None``.

    Attributes
    ----------
    sensor_array : LinearHydrophoneArray
        The towed hydrophone array, updated in-place on every :meth:`move`.

    """

    sensor_array: LinearHydrophoneArray = Property(doc="The towed hydrophone array.")
    cable_length_m: float = Property(
        doc="Slant distance in metres from the host to the first array element."
    )
    array_depth_m: float = Property(doc="Depth in metres at which the array is deployed.")
    velocity_mapping: Sequence[int] | None = Property(
        default=None,
        doc=(
            "Indices for velocity components in the state vector. "
            "Defaults to position_mapping + 1 when None."
        ),
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the platform and set initial array element states.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``MultiTransitionMovingPlatform``.
        **kwargs : object
            Keyword arguments forwarded to ``MultiTransitionMovingPlatform``.

        Raises
        ------
        ValueError
            If the platform has no initial state or if the cable length is
            geometrically incompatible with the array depth.

        """
        super().__init__(*args, **kwargs)

        if self.velocity_mapping is None:
            self._property_velocity_mapping = [p + 1 for p in self.position_mapping]

        if self.states:
            self._initialise_array(self.states[0].timestamp)

    def _resolved_velocity_mapping(self) -> Sequence[int]:
        """Return a guaranteed velocity mapping, falling back to position_mapping + 1."""
        if self.velocity_mapping is None:
            return [p + 1 for p in self.position_mapping]
        return self.velocity_mapping

    def _trail_direction_xy(self, host_state_vector: NDArray) -> NDArray:
        """Compute the 2-D unit vector pointing in the direction the array trails.

        Parameters
        ----------
        host_state_vector : ndarray
            Full host state vector.

        Returns
        -------
        ndarray
            Shape ``(2,)`` unit vector opposite to the host's horizontal velocity.
            Falls back to ``[-1, 0]`` when the host is stationary.

        """
        velocity_mapping = self._resolved_velocity_mapping()
        vel_xy = host_state_vector[velocity_mapping[:2]].flatten()
        vel_norm = float(np.linalg.norm(vel_xy))
        if vel_norm > 0:
            return -vel_xy / vel_norm
        return np.array([-1.0, 0.0])

    def _initialise_array(self, timestamp: datetime) -> None:
        """Compute initial array element positions from the host's first state.

        Parameters
        ----------
        timestamp : datetime
            Timestamp of the initial platform state.

        """
        sv = self.states[0].state_vector
        host_pos = sv[self.position_mapping].flatten()
        trail_dir = self._trail_direction_xy(sv)

        self.sensor_array.move(
            leader_pos=host_pos,
            trail_direction_xy=trail_dir,
            cable_length_m=self.cable_length_m,
            array_depth_m=self.array_depth_m,
            timestamp=timestamp,
        )

    def move(self, timestamp: datetime, **kwargs: object) -> None:
        """Advance the host and recompute array element positions.

        Parameters
        ----------
        timestamp : datetime
            Target timestamp.
        **kwargs : object
            Forwarded to the host's transition model.

        """
        self.movement_controller.move(timestamp, **kwargs)

        sv = self.state.state_vector
        host_pos = sv[self.position_mapping].flatten()
        trail_dir = self._trail_direction_xy(sv)

        self.sensor_array.move(
            leader_pos=host_pos,
            trail_direction_xy=trail_dir,
            cable_length_m=self.cable_length_m,
            array_depth_m=self.array_depth_m,
            timestamp=timestamp,
        )

    @property
    def host_path(self) -> FloatArray | None:
        """Return the complete position history of the host vehicle.

        Returns
        -------
        FloatArray or None
            Array of shape ``(N, D)`` where ``N`` is the number of timesteps
            and ``D`` is the spatial dimensionality.  Returns ``None`` if no
            states exist.

        """
        if not self.states:
            return None
        return np.array([s.state_vector[self.position_mapping].flatten() for s in self.states])

    @property
    def sensor_paths(self) -> list[FloatArray]:
        """Return the position history of each array element.

        Returns
        -------
        list of FloatArray
            One array per element, each of shape ``(N, 3)`` where ``N`` is
            the number of recorded states for that element.

        """
        paths = []
        for element in self.sensor_array.elements:
            if element.states:
                path = np.array([s.state_vector.flatten() for s in element.states])
            else:
                path = np.empty((0, 3))
            paths.append(path)
        return paths

    def __repr__(self) -> str:
        """Return a string representation of the platform configuration.

        Returns
        -------
        str
            Human-readable summary of key platform parameters.

        """
        return (
            f"TowedArrayPlatform("
            f"num_elements={self.sensor_array.num_elements}, "
            f"cable_length_m={self.cable_length_m}, "
            f"element_spacing_m={self.sensor_array.element_spacing_m}, "
            f"array_depth_m={self.array_depth_m})"
        )

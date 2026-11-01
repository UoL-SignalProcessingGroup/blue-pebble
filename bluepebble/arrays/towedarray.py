"""Defines a towed array: a line of sensors trailing behind a platform on a tow cable."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from stonesoup.base import Base, Property
from stonesoup.movable.movable import MovingMovable
from stonesoup.types.angle import mod_bearing
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import State

from .base import ArrayState, FloatArray, SensorArray

if TYPE_CHECKING:
    from stonesoup.movable.movable import Movable

class _StateHistoryCarrier(Protocol):
    """Protocol for objects exposing historical state sequences."""

    states: Sequence[State]


class _StateVectorCarrier(Protocol):
    """Protocol for objects exposing a direct state vector."""

    state_vector: StateVector


class _PositionCarrier(Protocol):
    """Protocol for objects exposing a live ``.position``.

    This is all a :class:`_FollowerModel` needs from its leader: a
    :class:`~stonesoup.movable.movable.MovingMovable` (a sensor following the one ahead) or the
    :class:`TowedArray` itself, whose mounted position is the tow point the first sensor
    follows.
    """

    position: StateVector


class _FollowerModel(Base):
    """A transition model that causes a movable to follow another movable.

    A generic model that causes a movable to follow a leader in 3D space and maintain a fixed 3D
    distance from the leader.

    Attributes
    ----------
    leader : MovingMovable or _PositionCarrier
        The leader that the follower will follow.
    offset : float
        The distance the follower should maintain from the leader in 3D space.

    """

    leader: "MovingMovable | _PositionCarrier" = Property(
        doc="The leader (a movable, or a virtual point such as a tow point) that this "
        "follower tracks."
    )
    offset: float = Property(doc="The distance the follower should maintain from the leader.")

    def function(self, state: State, **kwargs: object) -> StateVector:
        """Calculate the new 3D position of the follower.

        Parameters
        ----------
        state : State
            The current state of the follower, containing its position in the ``state_vector``.
        **kwargs
            Additional keyword arguments (present for TransitionModel compatibility; ignored by
            this implementation).

        Returns
        -------
        StateVector
            The new position of the follower, maintaining the specified offset from the leader.

        """
        follower_pos_old = state.state_vector
        leader_pos_new = self.leader.position
        vec_to_leader = leader_pos_new - follower_pos_old
        dist_to_leader = np.linalg.norm(vec_to_leader)
        if np.isclose(dist_to_leader, 0):
            direction_vec = np.array([[-1.0], [0.0], [0.0]])
        else:
            direction_vec = vec_to_leader / dist_to_leader
        new_position = leader_pos_new - self.offset * direction_vec
        return StateVector(new_position)


class _TowedArrayFollowerModel(_FollowerModel):
    """A specialised follower model for a towed array segment.

    This model enforces that the follower maintains a fixed depth. The ``offset`` property is
    interpreted as the slant distance between the leader and follower; horizontal separation is
    computed from the slant distance and vertical separation.

    Attributes
    ----------
    leader : MovingMovable
        The leader platform that the follower will follow.
    offset : float
        The distance the follower should maintain from the leader in the
        horizontal plane.
    array_depth_m : float
        The fixed depth at which the follower should be maintained.

    """

    array_depth_m: float = Property(
        doc="The fixed depth at which the follower should be maintained."
    )

    def function(self, state: State, **kwargs: object) -> StateVector:
        """Calculate the new position in 2D while keeping the depth fixed.

        The horizontal offset from the leader is computed using the Pythagorean theorem from the
        slant ``offset`` and the vertical separation to the desired ``array_depth_m``.

        Parameters
        ----------
        state : State
            The current state of the follower, containing its position in the ``state_vector``.
        **kwargs
            Additional keyword arguments (present for TransitionModel compatibility; ignored).

        Returns
        -------
        StateVector
            The new position of the follower, maintaining the fixed depth.

        """
        follower_pos_old = state.state_vector
        leader_pos_new = self.leader.position

        follower_pos_old_xy = follower_pos_old[:2]
        leader_pos_new_xy = leader_pos_new[:2]

        depth_difference = abs(leader_pos_new[2, 0] - self.array_depth_m)
        if self.offset**2 < depth_difference**2:
            horizontal_offset = 0
        else:
            horizontal_offset = np.sqrt(self.offset**2 - depth_difference**2)

        vec_to_leader_xy = leader_pos_new_xy - follower_pos_old_xy
        dist_to_leader_xy = np.linalg.norm(vec_to_leader_xy)
        if np.isclose(dist_to_leader_xy, 0):
            direction_vec_xy = np.array([[-1.0], [0.0]])
        else:
            direction_vec_xy = vec_to_leader_xy / dist_to_leader_xy

        new_position_xy = leader_pos_new_xy - horizontal_offset * direction_vec_xy

        new_position = np.vstack([new_position_xy, [[self.array_depth_m]]])
        return StateVector(new_position)


@dataclass
class HostState:
    """A container for the platform's state.

    Parameters
    ----------
    state : State
        The state of the platform towing the array.
    heading_rad : float
        The platform's direction of travel, in radians anticlockwise from +x, in
        ``[-pi, pi)``.

    """

    state: State
    heading_rad: float


@dataclass
class PlatformState:
    """A data class to hold the state of the platform and towed array at one timestamp.

    Parameters
    ----------
    timestamp : datetime
        The time at which this state is valid.
    host : HostState
        The state container for the platform towing the array.
    array : ArrayState
        The state container for the sensor array.

    """

    timestamp: datetime
    host: HostState
    array: ArrayState

    @property
    def position(self) -> StateVector:
        """Return the position of the platform towing the array.

        Returns
        -------
        StateVector
            The 3D position vector (x, y, z) of the platform.

        """
        return StateVector(self.host.state.state_vector[[0, 2, 4]])


class TowedArray(SensorArray):
    """A line of sensors towed behind a platform on a cable.

    Mount it on a :class:`~bluepebble.platform.SonarPlatform` with ``add_sensor``. Its
    ``mounting_offset`` is the tow point relative to the platform's reference point (for
    example ``[-10, 0, 0]`` for a tow point 10 m aft of a bow reference point); by default the
    tow point is the reference point itself.

    Each sensor follows the one ahead of it (the first follows the tow point) at a fixed cable
    length or sensor spacing, but is otherwise free to lag, which is what bends the array
    through a turn. The array's shape therefore depends on the platform's trajectory up to the
    current time, not just its instantaneous state, so the array keeps its own per-timestep
    history from the moment it is mounted.

    Parameters
    ----------
    num_sensors : int
        Number of sensors in the array.
    cable_length_m : float
        Length of the main tow cable in metres (distance from the tow point to the first
        sensor).
    sensor_spacing_m : float
        Spacing between subsequent sensors in metres.
    array_depth_m : float
        Depth at which the array is towed in metres.
    reference_sensor_idx : int, optional
        Index of the reference sensor.

    Attributes
    ----------
    towed_sensors : list[MovingMovable]
        The simulated sensors trailing the platform.
    platform_history : list[PlatformState]
        The platform and array's composite state at each timestamp since mounting.

    """

    num_sensors: int = Property(doc="Number of sensors in the array")
    cable_length_m: float = Property(doc="Length of the main tow cable in metres")
    sensor_spacing_m: float = Property(doc="Spacing between sensors in metres")
    array_depth_m: float = Property(doc="Depth at which the array is towed in metres")
    reference_sensor_idx: int = Property(default=0, doc="Index of the reference sensor")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the TowedArray.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        self.towed_sensors: list[MovingMovable] = []
        self.platform_history: list[PlatformState] = []
        # Timestamp -> index of its first entry in platform_history, so lookups need not scan it.
        self._platform_history_index: dict[datetime, int] = {}

    def _on_mounted(self) -> None:
        """Lay the sensors out astern of the tow point and record the first state."""
        self._initialise_sensor_array()
        self._capture_platform_state(self._controller.state.timestamp)

    def _initialise_sensor_array(self) -> None:
        """Initialise the towed sensor array's geometry and follower models.

        This places every sensor astern of the tow point along the platform's current velocity
        vector.

        Raises
        ------
        ValueError
            If the platform does not have a valid state, or if the cable/sensor spacing is
            physically impossible given the depth difference.
        AttributeError
            If a leader object does not expose a valid position.

        """
        try:
            platform_state = self._controller.state
            platform_vel_xy = platform_state.state_vector[self._velocity_mapping[:2]]
            tow_point_3d = cast(StateVector, self.position)  # not None once mounted
        except (IndexError, AttributeError, KeyError) as e:
            raise ValueError(
                f"Platform must have a state with accessible "
                f"state_vector and position/velocity mappings: {e}"
            ) from e

        vel_norm = np.linalg.norm(platform_vel_xy)
        if vel_norm > 0:
            backwards_heading_xy = -platform_vel_xy / vel_norm
        else:
            backwards_heading_xy = StateVector([[-1.0], [0.0]])

        towed_sensors = []
        leader_node: MovingMovable | TowedArray = self

        def get_position_from_object(obj: object) -> StateVector:
            """Extract position from a generic object."""
            states = getattr(obj, "states", None)
            if isinstance(states, Sequence) and len(states) > 0:
                history_obj = cast(_StateHistoryCarrier, obj)
                return history_obj.states[-1].state_vector

            state_vector = getattr(obj, "state_vector", None)
            if state_vector is not None:
                vector_obj = cast(_StateVectorCarrier, obj)
                return vector_obj.state_vector

            raise AttributeError(f"Object {obj} doesn't have accessible position")

        cumulative_horizontal_dist = 0.0
        for i in range(self.num_sensors):
            offset = self.cable_length_m if i == 0 else self.sensor_spacing_m

            follower_model = _TowedArrayFollowerModel(
                leader=leader_node, offset=offset, array_depth_m=self.array_depth_m
            )

            if leader_node is self:
                leader_pos_3d = tow_point_3d
            else:
                leader_pos_3d = get_position_from_object(leader_node)

            depth_difference = abs(leader_pos_3d[2, 0] - self.array_depth_m)

            if offset**2 < depth_difference**2:
                raise ValueError(
                    f"Segment length ({offset}m) is too short for depth difference "
                    f"({depth_difference}m)."
                )

            horizontal_separation = float(np.sqrt(offset**2 - depth_difference**2))
            cumulative_horizontal_dist += horizontal_separation

            displacement_xy = cumulative_horizontal_dist * backwards_heading_xy
            follower_init_pos_xy = tow_point_3d[[0, 1]] + displacement_xy

            follower_init_pos = StateVector(
                [
                    follower_init_pos_xy[0, 0],
                    follower_init_pos_xy[1, 0],
                    self.array_depth_m,
                ]
            )

            follower_init_state = GroundTruthState(
                follower_init_pos, timestamp=platform_state.timestamp
            )

            follower = MovingMovable(
                states=[follower_init_state],
                position_mapping=[0, 1, 2],
                transition_model=follower_model,
            )

            towed_sensors.append(follower)
            leader_node = follower

        self.towed_sensors = towed_sensors

    def _capture_platform_state(self, timestamp: datetime) -> None:
        """Capture and store the state of the platform and array at a timestamp.

        Parameters
        ----------
        timestamp : datetime
            The time at which to capture the state.

        """
        platform_state = self._platform_state_at(timestamp)
        if platform_state is None or not self.towed_sensors:
            return
        sensor_states = []
        for sensor in self.towed_sensors:
            sensor_state = self._state_at(sensor, timestamp)
            if sensor_state is None:
                return
            sensor_states.append(sensor_state)

        platform_vel_xy = platform_state.state_vector[self._velocity_mapping[:2]]
        # arctan2 returns +pi for due -x; mod_bearing wraps it to [-pi, pi) like Bearing.
        heading_rad = float(
            mod_bearing(np.arctan2(platform_vel_xy[1, 0], platform_vel_xy[0, 0]))
        )

        host_state_container = HostState(state=platform_state, heading_rad=heading_rad)

        positions = np.hstack([s.state_vector for s in sensor_states])
        # Sensor 0 is nearest the tow point, so the line's forward axis runs from the last
        # sensor to the first. A single sensor has no line, so fall back to the heading.
        forward = np.asarray(positions[:, 0] - positions[:, -1], dtype=float)
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 0:
            axis = forward / forward_norm
        else:
            axis = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])

        array_state_container = ArrayState(
            num_sensors=self.num_sensors,
            state_vector=positions,
            ref_state_vector=sensor_states[self.reference_sensor_idx].state_vector,
            axis=StateVector(axis),
        )

        self._platform_history_index.setdefault(timestamp, len(self.platform_history))
        self.platform_history.append(
            PlatformState(
                timestamp=timestamp,
                host=host_state_container,
                array=array_state_container,
            )
        )

    @staticmethod
    def _state_at(movable: "Movable", timestamp: datetime) -> State | None:
        """Return a movable's state at ``timestamp``, or None if it has none.

        ``move`` has just appended the state being captured, so the latest state is checked
        first; the full scan only runs when it does not match.
        """
        states = movable.states
        if states and states[-1].timestamp == timestamp:
            return states[-1]
        for state in states:
            if state.timestamp == timestamp:
                return state
        return None

    def _on_platform_moved(self, timestamp: datetime, **kwargs: object) -> None:
        """Advance every towed sensor and record the new combined state.

        Called by :meth:`~bluepebble.platform.SonarPlatform.move` once the platform itself
        has moved to ``timestamp``.

        Parameters
        ----------
        timestamp : datetime
            The timestamp the platform was just moved to.
        **kwargs : dict
            Additional arguments passed to the followers' transition models.

        """
        for sensor in self.towed_sensors:
            sensor.move(timestamp, **kwargs)

        self._capture_platform_state(timestamp)

    def get_platform_state_at(self, timestamp: datetime) -> PlatformState | None:
        """Get the combined platform and array state at a specific timestamp.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        PlatformState or None
            The platform state if found, otherwise None.

        """
        index = self._platform_history_index.get(timestamp)
        if index is not None and index < len(self.platform_history):
            state = self.platform_history[index]
            if state.timestamp == timestamp:
                return state
        # Not indexed, or platform_history was changed directly: fall back to a scan.
        for state in self.platform_history:
            if state.timestamp == timestamp:
                return state
        return None

    def get_host_state_at(self, timestamp: datetime) -> State | None:
        """Get the towing platform's state at a specific timestamp.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        State or None
            The platform state if found, otherwise None.

        """
        return self._platform_state_at(timestamp)

    def get_sensor_states_at(self, timestamp: datetime) -> list[GroundTruthState] | None:
        """Get the states of all towed sensors at a specific timestamp.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        list[GroundTruthState] or None
            A list of ground truth states for each sensor in order, or None if
            any sensor state is missing for the timestamp.

        """
        all_states = []
        for sensor in self.towed_sensors:
            found_state = None
            for state in sensor:
                if state.timestamp == timestamp:
                    found_state = state
                    break
            if found_state is None:
                return None
            all_states.append(found_state)
        return all_states

    @property
    def array(self) -> ArrayState:
        """Return the array state at the platform's latest timestamp.

        Raises
        ------
        ValueError
            If no array state has been captured at that timestamp.

        """
        state = self.get_platform_state_at(self._controller.state.timestamp)
        if state is None:
            raise ValueError("No towed array state was captured at the platform's latest time.")
        return state.array

    @property
    def host_path(self) -> FloatArray | None:
        """Get the complete path of the towing platform.

        Returns
        -------
        FloatArray | None
            Array of shape ``(N, D)`` containing platform-position history, where
            ``N`` is the number of time steps and ``D`` is spatial dimension.
            Returns ``None`` if no states exist.

        """
        controller = self._controller
        if not controller.states:
            return None
        position_mapping = controller.position_mapping
        return np.array(
            [state.state_vector[position_mapping].flatten() for state in controller.states]
        )

    @property
    def sensor_paths(self) -> list[FloatArray]:
        """Get sensor position paths as a list of arrays.

        Returns
        -------
        list[FloatArray]
            One array per sensor, each containing position history over time.

        """
        if not self.towed_sensors:
            return []
        paths = []
        for sensor in self.towed_sensors:
            if sensor.states:
                sensor_path = np.array([state.state_vector.flatten() for state in sensor.states])
                paths.append(sensor_path)
            else:
                paths.append(np.array([]).reshape(0, 3))
        return paths

    def __repr__(self) -> str:
        """Return string representation of the array.

        Returns
        -------
        str
            A string describing the configured array parameters.

        """
        return (
            f"TowedArray(num_sensors={self.num_sensors}, "
            f"cable_length_m={self.cable_length_m}, "
            f"sensor_spacing_m={self.sensor_spacing_m}, "
            f"array_depth_m={self.array_depth_m})"
        )

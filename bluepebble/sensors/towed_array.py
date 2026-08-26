"""Defines a towed array sensor: a linear array trailing behind a host on a tow cable."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Base, Property
from stonesoup.movable.movable import MovingMovable
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import State

from .base import ArrayState, Sensor

if TYPE_CHECKING:
    from ..platform.base import HostPlatform

FloatArray: TypeAlias = NDArray[np.float64]


class _StateHistoryCarrier(Protocol):
    """Protocol for objects exposing historical state sequences."""

    states: Sequence[State]


class _StateVectorCarrier(Protocol):
    """Protocol for objects exposing a direct state vector."""

    state_vector: StateVector


class _PositionCarrier(Protocol):
    """Protocol for objects exposing a live ``.position``.

    This is all a :class:`_FollowerModel` actually needs from its leader -- satisfied by a real
    :class:`~stonesoup.movable.movable.MovingMovable` (later elements following the one ahead)
    or by a virtual :class:`_TowPoint` (the first element following the host's tow point).
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
            direction_vec = np.array([-1.0, 0.0, 0.0])
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


class _TowPoint:
    """A virtual point astern of the host that the first towed element follows.

    Exposes only ``.position`` -- all a :class:`_FollowerModel` needs from a leader --
    computed as the host's current position, displaced ``offset_m`` behind it along the
    host's current heading. This keeps the actual tow attachment point (typically near the
    stern) distinct from the host's own reference point (typically the bow, where
    :class:`~bluepebble.sensors.bow_array.BowArraySensor` is centred), without needing a real
    stonesoup ``Movable``. Recomputed live on every access, so it tracks the host through
    turns rather than being fixed at its initial heading.
    """

    def __init__(self, host: "HostPlatform", offset_m: float) -> None:
        self._host = host
        self._offset_m = offset_m

    @property
    def position(self) -> StateVector:
        """Return the host's current position, displaced aft by ``offset_m``."""
        velocity_mapping = self._host.resolved_velocity_mapping()
        velocity_xy = self._host.state.state_vector[velocity_mapping[:2]]
        vel_norm = np.linalg.norm(velocity_xy)
        if vel_norm > 0:
            backwards_xy = -velocity_xy / vel_norm
        else:
            backwards_xy = StateVector([[-1.0], [0.0]])

        host_position = self._host.position
        return StateVector(
            [
                host_position[0, 0] + self._offset_m * backwards_xy[0, 0],
                host_position[1, 0] + self._offset_m * backwards_xy[1, 0],
                host_position[2, 0],
            ]
        )


@dataclass
class HostState:
    """A snapshot of the host vehicle's state, captured alongside an array state.

    Parameters
    ----------
    state : GroundTruthState
        The ground truth state of the host vehicle, taken directly from the host's own
        recorded history (not a copy).
    heading_rad : float
        The heading of the host vehicle in radians.

    """

    state: GroundTruthState
    heading_rad: float


@dataclass
class PlatformState:
    """A data class to hold the state of the host + towed array at one timestamp.

    Parameters
    ----------
    timestamp : datetime
        The time at which this state is valid.
    host : HostState
        The state container for the host vehicle.
    array : ArrayState
        The state container for the sensor array.

    """

    timestamp: datetime
    host: HostState
    array: ArrayState

    @property
    def position(self) -> StateVector:
        """Return the position of the host vehicle.

        Returns
        -------
        StateVector
            The 3D position vector (x, y, z) of the host vehicle.

        """
        return StateVector(self.host.state.state_vector[[0, 2, 4]])


class TowedArraySensor(Sensor):
    """A linear array of sensors towed behind a host on a cable.

    Each element follows the one ahead of it (or the host, for the first element) based on a
    defined cable length and sensor spacing, maintaining a fixed 3D distance but otherwise
    free to lag -- which is what produces the array's characteristic "bend" through a turn.
    That per-element lag means (unlike :class:`~bluepebble.sensors.bow_array.BowArraySensor`)
    the array's shape depends on the host's trajectory up to the current time, not just its
    instantaneous state, so this sensor keeps its own per-timestep history.

    Parameters
    ----------
    host : HostPlatform
        The platform this array is towed behind.
    num_sensors : int
        Number of sensors in the array.
    cable_length_m : float
        Length of the main tow cable in meters (distance from host to first sensor).
    sensor_spacing_m : float
        Spacing between subsequent sensors in meters.
    array_depth_m : float
        Depth at which the array is towed in meters.
    reference_sensor_idx : int, optional
        Index of the reference sensor. Defaults to 0.
    towed_array_bow_offset_m : float, optional
        Horizontal distance from the host's reference point (typically the bow, where
        :class:`~bluepebble.sensors.bow_array.BowArraySensor` is centred) to the tow point the
        cable actually starts from (typically the stern), in metres. Defaults to 10.0.

    Attributes
    ----------
    towed_sensors : list[MovingMovable]
        A list of the simulated sensor objects trailing the host.
    platform_history : list[PlatformState]
        A history of the host + array's composite state over time.

    """

    num_sensors: int = Property(doc="Number of sensors in the array")
    cable_length_m: float = Property(doc="Length of the main tow cable in meters")
    sensor_spacing_m: float = Property(doc="Spacing between sensors in meters")
    array_depth_m: float = Property(doc="Depth at which the array is towed in meters")
    reference_sensor_idx: int = Property(default=0, doc="Index of the reference sensor")
    towed_array_bow_offset_m: float = Property(
        default=10.0,
        doc="Horizontal distance from the host's reference point (typically the bow) to the "
        "tow point the cable actually starts from (typically the stern), in metres.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the TowedArraySensor.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)

        super(Base, self).__setattr__("platform_history", [])

        self._initialise_sensor_array()

        if self.host.states:
            self._capture_platform_state(self.host.states[0].timestamp)

        self.host.attach(self)

    def _initialise_sensor_array(self) -> None:
        """Initialise the towed sensor array's geometry and follower models.

        This sets up the initial positions of all sensors relative to the host based on the host's
        initial velocity vector.

        Raises
        ------
        ValueError
            If the host does not have a valid initial state, or if the
            cable/sensor spacing is physically impossible given the depth
            difference.
        AttributeError
            If a leader object does not expose a valid position.

        """
        try:
            velocity_mapping = self.host.resolved_velocity_mapping()
            host_state = self.host.states[0]
            host_pos_3d = host_state.state_vector[self.host.position_mapping]
            host_vel_xy = host_state.state_vector[velocity_mapping[:2]]
        except (IndexError, AttributeError, KeyError) as e:
            raise ValueError(
                f"Host must have an initial state with accessible "
                f"state_vector and position/velocity mappings: {e}"
            ) from e

        vel_norm = np.linalg.norm(host_vel_xy)
        if vel_norm > 0:
            backwards_heading_xy = -host_vel_xy / vel_norm
        else:
            backwards_heading_xy = StateVector([[-1.0], [0.0]])

        towed_sensors = []
        leader_node: MovingMovable | _TowPoint = _TowPoint(
            self.host, self.towed_array_bow_offset_m
        )

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

        cumulative_horizontal_dist = self.towed_array_bow_offset_m
        for i in range(self.num_sensors):
            offset = self.cable_length_m if i == 0 else self.sensor_spacing_m

            follower_model = _TowedArrayFollowerModel(
                leader=leader_node, offset=offset, array_depth_m=self.array_depth_m
            )

            if isinstance(leader_node, _TowPoint):
                leader_pos_3d = leader_node.position
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
            follower_init_pos_xy = host_pos_3d[[0, 1]] + displacement_xy

            follower_init_pos = StateVector(
                [
                    follower_init_pos_xy[0, 0],
                    follower_init_pos_xy[1, 0],
                    self.array_depth_m,
                ]
            )

            follower_init_state = GroundTruthState(
                follower_init_pos, timestamp=host_state.timestamp
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
        """Capture and store the state of the host + array at a timestamp.

        Parameters
        ----------
        timestamp : datetime
            The time at which to capture the state.

        """
        host_state = self.get_host_state_at(timestamp)
        sensor_states = self.get_sensor_states_at(timestamp)

        if not host_state or not sensor_states:
            return

        # Calculate heading from velocity
        velocity_mapping = self.host.resolved_velocity_mapping()
        host_vel_xy = host_state.state_vector[velocity_mapping[:2]]
        heading_rad = float(np.arctan2(host_vel_xy[1, 0], host_vel_xy[0, 0]))

        host_state_container = HostState(state=host_state, heading_rad=heading_rad)

        array_state_container = ArrayState(
            num_sensors=self.num_sensors,
            state_vector=np.hstack([s.state_vector for s in sensor_states]),
            ref_state_vector=sensor_states[self.reference_sensor_idx].state_vector,
        )

        self.platform_history.append(
            PlatformState(
                timestamp=timestamp,
                host=host_state_container,
                array=array_state_container,
            )
        )

    def _on_host_moved(self, timestamp: datetime, **kwargs: object) -> None:
        """Advance every towed follower and record the new combined state.

        Called by :meth:`~bluepebble.platform.base.HostPlatform.move` once the host itself
        has already moved to ``timestamp``.

        Parameters
        ----------
        timestamp : datetime
            The timestamp the host was just moved to.
        **kwargs : dict
            Additional arguments passed to the followers' transition models.

        """
        for sensor in self.towed_sensors:
            sensor.move(timestamp, **kwargs)

        self._capture_platform_state(timestamp)

    def get_platform_state_at(self, timestamp: datetime) -> PlatformState | None:
        """Get the combined host + array state at a specific timestamp.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        PlatformState or None
            The platform state if found, otherwise None.

        """
        for state in self.platform_history:
            if state.timestamp == timestamp:
                return state
        return None

    def get_host_state_at(self, timestamp: datetime) -> GroundTruthState | None:
        """Get the host vehicle's state at a specific timestamp.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        GroundTruthState or None
            The host state if found, otherwise None.

        """
        return self.host.get_state_at(timestamp)

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
    def host_path(self) -> FloatArray | None:
        """Get the complete path of the host vehicle.

        Returns
        -------
        FloatArray | None
            Array of shape ``(N, D)`` containing host-position history, where
            ``N`` is the number of time steps and ``D`` is spatial dimension.
            Returns ``None`` if no states exist.

        """
        if not self.host.states:
            return None
        position_mapping = self.host.position_mapping
        return np.array(
            [state.state_vector[position_mapping].flatten() for state in self.host.states]
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
                paths.append(np.array([]).reshape(0, len(self.host.position_mapping)))
        return paths

    def __repr__(self) -> str:
        """Return string representation of the sensor.

        Returns
        -------
        str
            A string describing the configured array parameters.

        """
        return (
            f"TowedArraySensor(num_sensors={self.num_sensors}, "
            f"cable_length_m={self.cable_length_m}, "
            f"sensor_spacing_m={self.sensor_spacing_m}, "
            f"array_depth_m={self.array_depth_m})"
        )

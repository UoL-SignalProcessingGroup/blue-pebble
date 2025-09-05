"""Defines a towed array platform.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.movable.movable import MovingMovable
from stonesoup.platform.base import MovingPlatform
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.groundtruth import GroundTruthState

from nereus.models.positioning import TowedArrayFollowerModel


@dataclass
class HostState:
    """A container for the host vehicle's state."""

    state: GroundTruthState


@dataclass
class ArrayState:
    """A container for the towed array's state and properties."""

    num_sensors: int
    state_vector: StateVectors
    ref_state_vector: StateVector


@dataclass
class PlatformState:
    """A data class to hold the state of the entire platform at one timestamp."""

    timestamp: datetime
    host: HostState
    array: ArrayState


class TowedArrayPlatform(MovingPlatform):
    """A Stone Soup compliant platform that can tow an array of sensors.

    This class models a towed array platform where the ship moves and the sensors follow
    along a cable. The sensors are positioned at regular intervals along the cable,
    maintaining a fixed distance from the ship.

    The platform inherits from MovingPlatform to provide proper Stone Soup integration
    including movement capabilities, sensor management, and state tracking.

    Attributes:
        num_sensors (int): Number of sensors in the array.
        cable_length_m (float): Length of the main tow cable in meters.
        sensor_spacing_m (float): Spacing between sensors in meters.
        array_depth_m (float): Depth at which the array is towed in meters.
        velocity_mapping (Sequence[int]): Indices for velocity in the state vector.
        reference_sensor_idx (int): Index of the reference sensor.
        towed_sensors (list[MovingMovable]): A list of follower platforms
            representing the towed sensor array (separate from platform-mounted
            sensors).

    """

    num_sensors = Property(int, doc="Number of sensors in the array")
    cable_length_m = Property(float, doc="Length of the main tow cable in meters")
    sensor_spacing_m = Property(float, doc="Spacing between sensors in meters")
    array_depth_m = Property(float, doc="Depth at which the array is towed in meters")
    velocity_mapping = Property(
        Sequence[int],
        default=None,
        doc="Indices for velocity in the state vector. If not set, defaults to "
        "position_mapping indices + 1",
    )
    reference_sensor_idx = Property(int, default=0, doc="Index of the reference sensor")

    def __init__(self, *args, **kwargs):
        """Initialise the TowedArrayPlatform.

        Args:
            *args: Positional arguments passed to MovingPlatform.
            **kwargs: Keyword arguments passed to MovingPlatform, including properties
                like `num_sensors`, `cable_length_m`, `sensor_spacing_m`,
                `array_depth_m`, `states`, `position_mapping`, and
                `transition_model`.

        """
        super().__init__(*args, **kwargs)

        # Set default velocity mapping if not provided
        if self.velocity_mapping is None:
            self._property_velocity_mapping = [p + 1 for p in self.position_mapping]

        super().__setattr__("platform_history", [])

        # Initialise after parent setup is complete
        self._initialise_sensor_array()

        if self.states:
            self._capture_platform_state(self.states[0].timestamp)

    def _initialise_sensor_array(self):
        """Initialise the towed sensor array's geometry and follower models.

        This method sets up the initial positions of all sensors in the towed
        array. It calculates a backwards heading from the host vehicle's
        initial velocity. It then iteratively creates each sensor as a
        `MovingMovable` with a `TowedArrayFollowerModel`, forming a
        leader-follower chain where each sensor follows the one ahead of it.
        The initial positions are calculated based on cable geometry, accounting
        for the depth difference between nodes.
        """
        # Get the host platform's initial state and position
        try:
            host_state = self.states[0]
            host_pos_3d = host_state.state_vector[self.position_mapping]
            host_vel_xy = host_state.state_vector[self.velocity_mapping[:2]]
        except (IndexError, AttributeError, KeyError) as e:
            raise ValueError(
                f"Platform must have an initial state with accessible "
                f"state_vector and position/velocity mappings: {e}"
            ) from e

        # Calculate backwards heading for sensor positioning
        vel_norm = np.linalg.norm(host_vel_xy)
        if vel_norm > 0:
            backwards_heading_xy = -host_vel_xy / vel_norm
        else:
            # Default backwards direction if no velocity
            backwards_heading_xy = StateVector([[-1.0], [0.0]])

        # Initialise towed sensors array (different name to avoid confusion with
        # platform sensors)
        towed_sensors = []
        leader_node = self.movement_controller  # Use the movement controller as leader
        cumulative_horizontal_dist = 0.0

        def get_position_from_object(obj):
            """Duck-typed position extraction."""
            if hasattr(obj, "states") and obj.states:
                return obj.states[-1].state_vector
            elif hasattr(obj, "state_vector"):
                return obj.state_vector
            else:
                raise AttributeError(f"Object {obj} doesn't have accessible position")

        for i in range(self.num_sensors):
            offset = self.cable_length_m if i == 0 else self.sensor_spacing_m

            # Create follower model for this sensor
            follower_model = TowedArrayFollowerModel(
                leader=leader_node, offset=offset, array_depth_m=self.array_depth_m
            )

            # Calculate sensor position using duck typing for state access
            if leader_node is self.movement_controller:
                leader_pos_3d = host_pos_3d
            else:
                leader_pos_3d = get_position_from_object(leader_node)

            # Calculate depth difference for cable geometry
            depth_difference = abs(leader_pos_3d[2, 0] - self.array_depth_m)

            if offset**2 < depth_difference**2:
                raise ValueError(
                    f"Segment length ({offset}m) is too short for depth difference "
                    f"({depth_difference}m)."
                )

            # Calculate horizontal separation
            horizontal_separation = np.sqrt(offset**2 - depth_difference**2)
            cumulative_horizontal_dist += horizontal_separation

            # Position sensor behind the ship
            displacement_xy = cumulative_horizontal_dist * backwards_heading_xy
            follower_init_pos_xy = host_pos_3d[[0, 1]] + displacement_xy

            follower_init_pos = StateVector(
                [
                    follower_init_pos_xy[0, 0],
                    follower_init_pos_xy[1, 0],
                    self.array_depth_m,
                ]
            )

            # Create sensor as MovingMovable
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

    def _capture_platform_state(self, timestamp: datetime):
        """Capture and store the state of the entire platform at a timestamp."""
        host_state = self.get_host_state_at(timestamp)
        sensor_states = self.get_sensor_states_at(timestamp)

        # Abort if any state information is missing for this timestamp
        if not host_state or not sensor_states:
            return

        # Create the HostState
        host_state_container = HostState(state=host_state)

        # Create the ArrayState (pre-calculating useful properties)
        array_state_container = ArrayState(
            num_sensors=self.num_sensors,
            state_vector=np.hstack([s.state_vector for s in sensor_states]),
            ref_state_vector=sensor_states[self.reference_sensor_idx].state_vector,
        )

        # Create the final PlatformState and append to history
        self.platform_history.append(
            PlatformState(
                timestamp=timestamp,
                host=host_state_container,
                array=array_state_container,
            )
        )

    def move(self, timestamp: datetime, **kwargs) -> None:
        """Move the platform and all sensor followers.

        Args:
            timestamp: The timestamp to move to.
            **kwargs: Additional arguments passed to the movement methods.

        """
        # Move the platform itself via its movement controller
        self.movement_controller.move(timestamp, **kwargs)

        # Move all towed sensors in the array
        for sensor in self.towed_sensors:
            sensor.move(timestamp, **kwargs)

        self._capture_platform_state(timestamp)

    def get_platform_state_at(self, timestamp: datetime) -> PlatformState | None:
        """Get the platform state at a specific timestamp.

        Args:
            timestamp: The timestamp to retrieve the state for.

        Returns:
            PlatformState: The platform state at the given timestamp, or None if not "
            "found.

        """
        for state in self.platform_history:
            if state.timestamp == timestamp:
                return state
        return None

    def get_host_state_at(self, timestamp: datetime) -> GroundTruthState | None:
        """Get the host vehicle's state at a specific timestamp."""
        for state in self.movement_controller:
            if state.timestamp == timestamp:
                return state
        return None

    def get_sensor_states_at(
        self, timestamp: datetime
    ) -> list[GroundTruthState] | None:
        """Get the states of all towed sensors at a specific timestamp.

        Args:
            timestamp: The timestamp to get the states at.

        Returns:
            A list of GroundTruthState objects for all sensors at the given
            timestamp, or None if any sensor state is not found.

        """
        all_states = []
        for sensor in self.towed_sensors:
            found_state = None
            for state in sensor:
                if state.timestamp == timestamp:
                    found_state = state
                    break
            if found_state is None:
                return None  # If any sensor is missing a state, abort
            all_states.append(found_state)
        return all_states

    @property
    def host_path(self):
        """Get the complete path of the host vehicle.

        Returns:
            np.ndarray: Shape (n_timesteps, n_dims) containing host vehicle positions
                over time. Returns None if no states available.

        """
        if not self.states:
            return None

        return np.array(
            [
                state.state_vector[self.position_mapping].flatten()
                for state in self.states
            ]
        )

    @property
    def sensor_paths(self):
        """Get sensor position paths as a list of arrays.

        Returns:
            list[np.ndarray]: List where each element is a (n_timesteps, n_dims)
                array containing one sensor's position history. Returns empty list
                if no sensors available.

        """
        if not self.towed_sensors:
            return []

        paths = []
        for sensor in self.towed_sensors:
            if sensor.states:
                # Extract positions only (not full state vector)
                sensor_path = np.array(
                    [state.state_vector.flatten() for state in sensor.states]
                )
                paths.append(sensor_path)
            else:
                # Handle sensor with no states
                paths.append(np.array([]).reshape(0, len(self.position_mapping)))

        return paths

    def __repr__(self):
        """Return string representation of the platform."""
        return (
            f"TowedArrayPlatform(num_sensors={self.num_sensors}, "
            f"cable_length_m={self.cable_length_m}, "
            f"sensor_spacing_m={self.sensor_spacing_m}, "
            f"array_depth_m={self.array_depth_m})"
        )

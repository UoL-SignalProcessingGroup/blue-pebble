"""Defines a towed array platform.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from datetime import datetime

import numpy as np
from stonesoup.models.transition.base import TransitionModel
from stonesoup.movable import MovingMovable
from stonesoup.types.groundtruth import GroundTruthState

from nereus.stonesoup.transition import TowedArrayFollowerModel


class TowedArrayPlatform:
    """A platform that can tow an array of sensors.

    This class models a towed array platform where the ship moves and the sensors follow
    along a cable. The sensors are positioned at regular intervals along the cable,
    maintaining a fixed distance from the ship.
    The ship's movement is defined by a transition model, and the sensors follow the
    ship with a specified offset.
    The array is towed at a specified depth, independent of the ship's depth.
    The ship's position is updated based on its transition model, while the sensors
    are updated to maintain their positions relative to the ship.

    Attributes:
        ship (MovingMovable): The leader platform representing the ship.
        sensors (list[MovingMovable]): A list of follower platforms representing the
            sensors.

    """

    def __init__(
        self,
        state: GroundTruthState,
        position_mapping: list[int],
        velocity_mapping: list[int],
        transition_model: TransitionModel,
        num_sensors: int,
        cable_length: float,
        sensor_spacing: float,
        array_depth: float,
    ) -> None:
        """Initialise the TowedArrayPlatform.

        Args:
            state (GroundTruthState): Initial state of the leader platform.
            position_mapping (list): Indices for position in the state vector.
            velocity_mapping (list): Indices for velocity in the state vector.
            transition_model (TransitionModel): Transition model for the leader.
            num_sensors (int): Number of sensors in the array.
            cable_length (float): Length of the main tow cable.
            sensor_spacing (float): Spacing between sensors along the cable.
            array_depth (float): Depth at which the array is towed.

        """
        ship = MovingMovable(
            states=[state],
            position_mapping=position_mapping,
            transition_model=transition_model,
        )

        sensors = []
        ship_pos_3d = ship.states[0].state_vector[position_mapping]
        ship_vel_xy = ship.states[0].state_vector[velocity_mapping]
        vel_norm = np.linalg.norm(ship_vel_xy)
        if vel_norm > 0:
            backwards_heading_xy = -ship_vel_xy / vel_norm
        else:
            backwards_heading_xy = np.array([[-1.0], [0.0]])

        leader_node = ship
        cumulative_horizontal_dist = 0.0

        for i in range(num_sensors):
            offset = cable_length if i == 0 else sensor_spacing
            follower_model = TowedArrayFollowerModel(
                leader=leader_node, offset=offset, array_depth=array_depth
            )

            if leader_node is ship:
                # The leader is the ship, which has a 6D state vector
                leader_position_map = position_mapping
            else:
                # The leader is another follower, which has a 3D state vector
                leader_position_map = [0, 1, 2]

            leader_pos_3d = leader_node.states[0].state_vector[leader_position_map]

            depth_difference = abs(leader_pos_3d[2, 0] - array_depth)

            if offset**2 < depth_difference**2:
                raise ValueError(
                    f"Segment length ({offset}m) is too short for depth difference "
                    f"({depth_difference}m)."
                )
            # The horizontal distance for THIS segment
            horizontal_separation = np.sqrt(offset**2 - depth_difference**2)
            cumulative_horizontal_dist += horizontal_separation

            # Calculate total displacement from the SHIP in the XY plane
            displacement_xy = cumulative_horizontal_dist * backwards_heading_xy
            follower_init_pos_xy = ship_pos_3d[[0, 1]] + displacement_xy

            follower_init_pos = np.array(
                [
                    follower_init_pos_xy[0, 0],
                    follower_init_pos_xy[1, 0],
                    array_depth,
                ]
            ).reshape(-1, 1)
            # --- End Fix ---

            follower_init_state = GroundTruthState(
                follower_init_pos, timestamp=state.timestamp
            )
            follower = MovingMovable(
                states=[follower_init_state],
                position_mapping=[0, 1, 2],
                transition_model=follower_model,
            )
            sensors.append(follower)
            leader_node = follower

        self.ship = ship
        self.sensors = sensors
        self.array_depth = array_depth

    def move(self, timestamp: datetime, noise=False) -> None:
        """Move the leader and all followers to their new positions.

        This method updates the ship's position based on its transition model,
        and then updates each follower's position to maintain the correct offset
        from the ship.

        Args:
            timestamp (datetime): The current timestamp for the movement update.
            noise (bool): Whether to apply noise to the movement. Defaults to False.

        Note:
            The followers do not have noise applied to their movements.

        """
        self.ship.move(timestamp, noise=True)
        for sensor in self.sensors:
            sensor.move(timestamp)

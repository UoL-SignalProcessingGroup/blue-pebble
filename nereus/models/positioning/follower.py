"""Defines transition models for follower platforms.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np
from stonesoup.base import Base, Property
from stonesoup.movable.movable import MovingMovable
from stonesoup.types.array import StateVector
from stonesoup.types.state import State


class FollowerModel(Base):
    """A transition model that causes a movable to follow another movable.

    A generic model that causes a movable to follow a leader in 3D space.
    It maintains a fixed 3D distance from the leader.

    Attributes:
        leader (MovingMovable): The leader platform that the follower will follow.
        offset (float): The distance the follower should maintain from the leader in
            3D space.

    """

    leader = Property(
        MovingMovable, doc="The leader movable that the next movable will follow."
    )
    offset = Property(
        float, doc="The distance the follower should maintain from the leader."
    )

    def function(self, state: State, **kwargs) -> StateVector:
        """Calculate the new 3D position of the follower.

        Args:
            state (State): The current state of the follower, containing its
                position in the `state_vector`.
            **kwargs: Additional keyword arguments. Only used for compatibility with
                the TransitionModel interface.

        Returns:
            StateVector: The new position of the follower, maintaining the specified
                offset from the leader.

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


class TowedArrayFollowerModel(FollowerModel):
    """A specialised follower model for a towed array segment.

    This model overrides the base behavior to enforce that the follower
    maintains a fixed depth. The `offset` property is treated as the total
    slant distance between the leader and the follower. The horizontal
    separation is calculated based on this slant distance and the difference
    in depth, ensuring the follower remains on the correct XY position relative
    to the leader while holding its specified depth.

    Attributes:
        leader (MovingMovable): The leader platform that the follower will follow.
        offset (float): The distance the follower should maintain from the leader in
            the horizontal plane.
        array_depth_m (float): The fixed depth at which the follower should be
            maintained.

    """

    array_depth_m = Property(
        float, doc="The fixed depth at which the follower should be maintained."
    )

    def function(self, state: State, **kwargs) -> StateVector:
        """Calculate the new position in 2D while keeping the depth fixed.

        This method calculates the required horizontal offset from the leader
        using the Pythagorean theorem, based on the total `offset` (slant range)
        and the vertical separation between the leader and the target array depth.

        Args:
            state (State): The current state of the follower, containing its
                position in the `state_vector`.
            **kwargs: Additional keyword arguments. Only used for compatibility with
                the TransitionModel interface.

        Returns:
            StateVector: The new position of the follower, maintaining a fixed depth.

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

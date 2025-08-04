from datetime import timedelta

import numpy as np
from stonesoup.models.transition.base import TransitionModel
from stonesoup.movable import MovingMovable
from stonesoup.types.groundtruth import GroundTruthState


class FollowerModel(TransitionModel):
    """A transition model that causes a movable to follow another movable (the "leader")."""

    # This model defines a 3D position [x, y, z]
    ndim_state = 3

    def __init__(self, leader: MovingMovable, offset: float, *args, **kwargs):
        """Initialise the FollowerModel.

        Args:
            leader (MovingMovable): The movable object to follow.
            offset (float): The distance to maintain behind the leader.
            *args: Positional arguments passed to the parent TransitionModel.
            **kwargs: Keyword arguments passed to the parent TransitionModel.

        """
        super().__init__(*args, **kwargs)
        self.leader = leader
        self.offset = offset

    def function(
        self, state: GroundTruthState, time_interval: timedelta, **kwargs
    ) -> np.ndarray:
        """Calculate the new position of the follower.

        Note: We assume the leader has already been moved to the new timestamp.
        """
        follower_pos_old = state.state_vector

        # Get the leader's position at the new time.
        leader_pos_new = self.leader.position

        # Calculate the vector from the follower's old position to the leader's new one.
        vec_to_leader = leader_pos_new - follower_pos_old
        dist_to_leader = np.linalg.norm(vec_to_leader)

        # Avoid division by zero if they are in the same spot
        if np.isclose(dist_to_leader, 0):
            # Default to being straight behind on the x-axis
            direction_vec = np.array([-1.0, 0.0, 0.0])
        else:
            direction_vec = vec_to_leader / dist_to_leader

        # The new position is the leader's position, displaced backwards by the cable length.
        new_position = leader_pos_new - self.offset * direction_vec
        return new_position

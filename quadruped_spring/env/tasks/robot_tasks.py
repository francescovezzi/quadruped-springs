import numpy as np

from quadruped_spring.env.tasks.task_base import TaskJumping


class JumpingOnPlaceHeight(TaskJumping):
    """
    Robot has to perform one single jump on place. It has to fall the closest as possible
    to the place it landed. Sparse reward based on maximizing the absolute reached height.
    """

    def __init__(self):
        super().__init__()

    def _reward(self):
        """Remember the reward is sparse. So is 0 except the end of the episode."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        if self._terminated():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        max_height = 0.4
        max_height_normalized = self._relative_max_height / max_height
        reward += max_height_normalized
        reward += max_height_normalized * 0.05 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += max_height_normalized * 0.05 * np.exp(-self._max_roll**2 / 0.01)  # orientation

        reward += max_height_normalized * 0.1 * np.exp(-self._max_forward_distance**2 / 0.05)  # be on place

        if self._relative_max_height > 0 and not self._terminated():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * max_height_normalized
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward


class JumpingForward(TaskJumping):
    """
    Robot has to perform a forward jumping. Sparse reward based on maximizing the maximum flight time
    and the forward distance. Bonus for mantaining the right orientation, malus on crushing.
    """

    def __init__(self):
        super().__init__()

    def _reward(self):
        """Remember the reward is sparse. So is 0 except the end of the episode."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        if self._terminated():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        reward += self._max_flight_time
        max_distance = 0.2
        # Normalize forward distance reward
        reward += 0.1 * self._max_forward_distance / max_distance
        if self._max_flight_time > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * self._max_flight_time
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward
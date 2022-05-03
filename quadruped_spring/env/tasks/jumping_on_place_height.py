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
    
    def _reward_end_episode(self, reward):
        """Add bonus and malus at the end of the episode for jumping on place task"""
        if self._task_terminated():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        max_height = 0.4
        max_height_normalized = self._relative_max_height / max_height
        reward += max_height_normalized
        reward += max_height_normalized * 0.05 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += max_height_normalized * 0.05 * np.exp(-self._max_pitch**2 / 0.01)  # orientation

        reward += max_height_normalized * 0.1 * np.exp(-self._max_forward_distance**2 / 0.05)  # be on place

        if self._relative_max_height > 0 and not self._task_terminated():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * max_height_normalized
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward
import numpy as np

from quadruped_spring.env.tasks.task_base import TaskJumping


class JumpingOnPlaceHeight(TaskJumping):
    """
    Robot has to perform one single jump in place. It has to fall the closest as possible
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

        max_height = 0.1
        # max_height_normalized = self._relative_max_height / max_height
        if self._relative_max_height > max_height:
            max_height_normalized = 1.0
        else:
            max_height_normalized = self._relative_max_height / max_height
        reward += 0.8 * max_height_normalized
        reward += max_height_normalized * 0.03 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += max_height_normalized * 0.03 * np.exp(-self._max_roll**2 / 0.01)  # orientation

        reward += max_height_normalized * 0.05 * np.exp(-self._max_forward_distance**2 / 0.05)  # be on place

        reward += max_height_normalized * 0.08 * np.exp(-self._max_vel_err**2 / 0.001)  # vel direction is similar to [0,0,1]

        if not self._terminated():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * max_height_normalized
        else:
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08 * (1 + 0.1 * max_height_normalized)
        return reward

    def _reset(self, env):
        super()._reset(env)
        landing_pose = self._env._robot_config.INIT_MOTOR_ANGLES
        self._env._ac_interface.set_landing_pose(landing_pose)


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
            reward -= 0.08 * (1 + 0.1 * self._max_flight_time)

        max_distance = 0.2
        max_fwd_distance_normalized = self._max_forward_distance / max_distance

        reward += self._max_flight_time
        reward += 0.1 * max_fwd_distance_normalized

        reward += self._max_flight_time * 0.05 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += self._max_flight_time * 0.05 * np.exp(-self._max_roll**2 / 0.01)  # orientation

        if self._max_flight_time > 0 and not self._terminated():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * self._max_flight_time
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward


class JumpingInPlaceDense(TaskJumping):
    """
    Robot has to perform one single jump in place. It has to fall the closest as possible
    to the place it landed. Sparse reward based on maximizing the absolute reached height.
    """

    def __init__(self):
        super().__init__()

    def _reward(self):
        """Reward calculated each step."""
        lin_vel = self._env.robot.GetBaseLinearVelocity()[[0, 2]]
        _, pitch_rate, _ = self._env.robot.GetTrueBaseRollPitchYawRate()
        vel_ref = self._env._robot_sensors.get_desired_velocity()
        track_err = lin_vel - vel_ref
        acc = self._compute_base_acc()
        delta_action = self._compute_delta_action()

        pitch_rate_reward = -0.02 * np.abs(pitch_rate)
        err_reward = 1.0 * np.exp(-np.dot(track_err, track_err) / 0.02**2)
        acc_reward = -0.0005 * np.dot(acc, acc)
        action_reward = -0.00002 * np.dot(delta_action, delta_action)

        # reward = pitch_rate_reward + err_reward + acc_reward + action_reward
        reward = err_reward
        return reward

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        # if self._terminated():
        #     # Malus for crashing
        #     # Optionally: no reward in case of crash
        #     reward -= 0.04
        # else:
        #     reward += 0.04
        return reward

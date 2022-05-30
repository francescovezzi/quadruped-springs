import numpy as np

from quadruped_spring.env.tasks.task_base import TaskJumping


class JumpingOnPlaceHeight(TaskJumping):
    """
    Robot has to perform one single jump in place. It has to fall the closest as possible
    to the place it landed. Sparse reward based on maximizing the absolute reached height.
    """

    def __init__(self):
        super().__init__()
        self._height_min = 0.35
        self._height_max = 0.7
        self._max_height_task = self._compute_max_height_task()
        
    def on_curriculum_step(self, verbose):
        super().on_curriculum_step(verbose=verbose)
        self._max_height_task = self._compute_max_height_task()
        if verbose > 0:
            print(f'-- max height set to {self._max_height_task:.3f} --')

    def _reward(self):
        """Reward for each simulation step."""
        reward = 0
        if not self._switched_controller:  # Only if the robot is using RL policy
            # Penalize height base decreasing
            height = self._env.robot.GetBasePosition()[2]
            if self._init_height - height > 0.02:
                reward -= 0.008
            # Penalize high frequency torques command
            # tau_max = 300
            # delta_tau = self._new_torque - self._old_torque
            # delta_tau_module = np.sum(delta_tau**2)
            # if delta_tau_module > tau_max:
            #     reward -= 0.005 * delta_tau_module / 500
        return reward

    def _compute_max_height_task(self):
        """Compute the maximum robot base height desired for the task."""
        curr_level = self.get_curriculum_level()
        return self._height_min * (1 - curr_level) + self._height_max * curr_level

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0

        # Task reward -> Height
        max_height = self._max_height_task
        if self._relative_max_height > max_height:
            max_height_normalized = 1.0
        else:
            max_height_normalized = self._relative_max_height / max_height
        reward += 0.8 * max_height_normalized

        # Orientation -> Maintain the initial orientation if you can !
        reward += max_height_normalized * 0.02 * np.exp(-self._max_yaw**2 / 0.15**2)  # orientation
        reward += max_height_normalized * 0.02 * np.exp(-self._max_roll**2 / 0.15**2)  # orientation
        reward += max_height_normalized * 0.1 * np.exp(-self._max_pitch**2 / 0.15**2)  # orientation

        # Position -> jump in place !
        # reward += max_height_normalized * 0.05 * np.exp(-self._max_forward_distance**2 / 0.05)
        # reward += max_height_normalized * 0.02 * np.exp(-self._max_delta_x**2 / 0.1**2)
        # reward += max_height_normalized * 0.05 * np.exp(-self._max_delta_y**2 / 0.1**2)

        # Velocity -> velocity direction close to [0,0,1]
        # reward += max_height_normalized * 0.01 * np.exp(-self._max_vel_err**2 / 0.1**2)

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


class JumpingForwardHeight(TaskJumping):
    """
    Robot has to perform a forward jumping. Sparse reward based on maximizing the max height
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
        weight_matrix = np.array([[0.1, 0.0], [0.0, 1.0]])
        pitch_rate_reward = -0.002 * np.abs(pitch_rate)
        # err_reward = 1.0 * np.exp(-(track_err @  weight_matrix @ track_err) / 0.01**2)
        err_reward = -(track_err @ weight_matrix @ track_err)

        reward = err_reward  # + pitch_rate_reward

        height = self._env.robot.GetBasePosition()[2]
        if self._init_height - height > 0.02:
            reward -= -0.04

        return reward

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        return reward

    def _terminated(self):
        done = super()._terminated()
        if self._env.robot.GetBasePosition()[2] >= 0.36:
            done = True
        return done

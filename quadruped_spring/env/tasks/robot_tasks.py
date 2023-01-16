import os

import numpy as np

import quadruped_spring as qs
from quadruped_spring.env.tasks.task_base import TaskBase, TaskJumping, TaskJumpingDemo


class JumpingInPlace(TaskJumping):
    """
    Robot has to perform one single jump in place. It has to fall the closest as possible
    to the place it landed. Sparse reward based on maximizing the absolute reached height.
    """

    def __init__(self, env):
        super().__init__(env)
        self._max_height_task = 0.9

    def _reward(self):
        """Reward for each simulation step."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0

        # Task reward -> Height
        max_height = self._max_height_task
        if self._relative_max_height > max_height:
            max_height_normalized = 1.0
        else:
            max_height_normalized = self._relative_max_height / max_height
        reward += 0.7 * max_height_normalized

        # Orientation -> Maintain the initial orientation if you can !
        reward += max_height_normalized * 0.3 * np.exp(-self._max_pitch**2 / 0.15**2)  # orientation

        # Position -> jump in place !
        # reward += max_height_normalized * 0.05 * np.exp(-self._max_forward_distance**2 / 0.05)
        reward += max_height_normalized * 0.05 * np.exp(-self._max_delta_x**2 / 0.05)

        if not self._terminated():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * max_height_normalized
        else:
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08 * (1 + 0.8 * max_height_normalized)
        return reward


class JumpingForward(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self._max_height_task = 0.3
        self._max_forward_distance_task = 1.3

    def _reward(self):
        """Reward for each simulation step."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0

        max_height = self._max_height_task
        if self._relative_max_height > max_height:
            max_height_normalized = 1.0
        else:
            max_height_normalized = self._relative_max_height / max_height

        max_distance = self._max_forward_distance_task
        # print(f'max distance -> {self._max_forward_distance}')
        if self._max_forward_distance > self._max_forward_distance_task:
            max_fwd_distance_normalized = 1.0
        else:
            max_fwd_distance_normalized = self._max_forward_distance / max_distance

        bonus_malus_term = (max_height_normalized + max_fwd_distance_normalized) / 2

        reward += 0.25 * max_height_normalized
        reward += 0.5 * max_fwd_distance_normalized * max_height_normalized

        reward += max_height_normalized * 0.25 * np.exp(-self._max_pitch**2 / 0.15**2)  # orientation

        if not self._terminated():
            reward += 0.1 * bonus_malus_term
        else:
            reward -= 0.08 * (1 + 1.2 * bonus_malus_term)

        return reward


class NoTask(TaskBase):
    """No tasks is required to be performed. Useful for using TORQUE action interface."""

    def __init__(self, env=None):
        super().__init__(env)


class JumpingDemoInPlace(TaskJumpingDemo):
    def __init__(self, env):
        self.demo_path = os.path.join(os.path.dirname(qs.__file__), "demonstrations", "demo_list_jip_0.npy")
        super().__init__(env)


class JumpingDemoForward(TaskJumpingDemo):
    def __init__(self, env):
        self.demo_path = os.path.join(os.path.dirname(qs.__file__), "demonstrations", "demo_list_jf_0.npy")
        super().__init__(env)


class JumpingInPlacePPO(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self.rest_mode = False
        self.aci = self._env.get_ac_interface()
        self._init_reward_constants()
        self._init_reward_parameters()

    def _init_reward_constants(self):
        self.max_height_task = 1.0  # absolute [m]
        self.min_height_task = 0.29  # absolute [m]
        self.max_contact_force = 800  # [N]

    @staticmethod
    def norm(vec):
        return np.dot(vec, vec)

    def _init_reward_parameters(self):
        self.k_h = 0.023
        self.k_tau_sigma = 0.1
        self.k_tau = 0.015
        self.k_contact = 3e-4
        self.k_pos_sigma = 40.0
        self.k_pos = 0.013
        self.k_action_sigma = 0.1
        self.k_action = 0.1
        self.k_config_smoothing_sigma = 5
        self.k_config_smoothing = 0.025
        self.k_pitch_sigma = 26
        self.k_pitch = 0.014

    def get_clipped_height(self):
        actual_height = self._pos_abs[2]
        if actual_height < self.min_height_task:
            return 0
        elif actual_height > self.max_height_task:
            return 0
        else:
            return actual_height

    def get_over_contact_force(self):
        _, _, feet_forces, _ = self._env.robot.GetContactInfo()
        contact_force = np.sum(feet_forces)
        if contact_force > self.max_contact_force:
            return contact_force
        else:
            return 0

    def _reward_height(self):
        return self.k_h * self.get_clipped_height()

    def _reward_in_place(self):
        return self.k_pos * np.exp(-self.k_pos_sigma * np.abs(self._pos_abs[0]))

    def _reward_smoothing(self):
        actual_delta_torques = np.sqrt(self.norm(self._get_torque_diff()))
        return self.k_tau * np.exp(-self.k_tau_sigma * actual_delta_torques)

    def _reward_force_contact(self):
        return -self.k_contact * self.get_over_contact_force()

    def _reward_smooth_action(self):
        actual_delta_action = np.sqrt(self.norm(self._get_action_diff())) / 24
        return self.k_action * np.exp(-self.k_action_sigma * actual_delta_action)

    def _reward_config_smooth(self):
        config_des = self.aci._transform_action_to_motor_command(self._old_action)
        actual_config = self._env.robot.GetMotorAngles()
        delta_config = actual_config - config_des
        config_diff = np.sqrt(self.norm(delta_config))
        return self.k_config_smoothing * np.exp(-self.k_config_smoothing_sigma * config_diff)

    def _reward_pitch(self):
        return self.k_pitch * np.exp(-self.k_pitch_sigma * np.abs(self._orient_rpy[1]))

    def _aux_rew(self):
        self.rew_h += self._reward_height()
        self.rew_smooth += self._reward_smoothing()
        self.rew_contact += self._reward_force_contact()
        self.rew_pos += self._reward_in_place()
        self.rew_action += self._reward_smooth_action()
        self.rew_config += self._reward_config_smooth()
        self.rew_pitch += self._reward_pitch()

    def _reward(self):
        rew = 0
        rew_h = self._reward_height()
        rew_smooth = self._reward_smoothing()
        rew_contact = self._reward_force_contact()
        rew_pos = self._reward_in_place()
        rew_pitch = self._reward_pitch()

        rew += 0.05 * rew_pos + 0.5 * rew_contact + 0.2 * rew_smooth + 0.45 * rew_h + 0.3 * rew_pitch

        return rew

    def enable_rest_mode(self):
        self.rest_mode = True

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0

        if not self._terminated():
            reward += 0.0
        else:
            reward -= 0.25 * self._max_height

        return reward


class JumpingForwardPPO(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self.rest_mode = False
        self.aci = self._env.get_ac_interface()
        self._init_reward_constants()
        self._init_reward_parameters()

    def _init_reward_constants(self):
        self.max_height_task = 0.9  # absolute [m]
        self.min_height_task = 0.29  # absolute [m]
        self.max_contact_force = 800  # [N]
        self.max_fwd = 1.3

    @staticmethod
    def norm(vec):
        return np.dot(vec, vec)

    def _init_reward_parameters(self):
        self.k_h = 0.026
        self.k_tau_sigma = 0.1
        self.k_tau = 0.015
        self.k_contact = 3e-4
        self.k_pos_sigma = 40.0
        self.k_pos = 0.013
        self.k_action_sigma = 0.1
        self.k_action = 0.1
        self.k_config_smoothing_sigma = 5
        self.k_config_smoothing = 0.025
        self.k_pitch_sigma = 26
        self.k_pitch = 0.014
        self.k_fwd = 0.038

    def get_clipped_height(self):
        actual_height = self._pos_abs[2]
        if actual_height < self.min_height_task:
            return 0
        elif actual_height > self.max_height_task:
            return 0
        else:
            return actual_height

    def get_over_contact_force(self):
        _, _, feet_forces, _ = self._env.robot.GetContactInfo()
        contact_force = np.sum(feet_forces)
        if contact_force > self.max_contact_force:
            return contact_force
        else:
            return 0

    def get_fwd(self):
        fwd = self.actual_fwd
        if fwd > self.max_fwd or fwd == self.old_fwd:
            return 0
        else:
            return fwd

    def _reset(self):
        self.old_fwd = 0.0
        self.actual_fwd = 0.0
        super()._reset()

    def update_fwd(self):
        self.old_fwd = self.actual_fwd
        self.actual_fwd = self._max_forward_distance

    def _on_step(self):
        super()._on_step()
        self.update_fwd()

    def _reward_height(self):
        return self.k_h * self.get_clipped_height()

    def _reward_in_place(self):
        return self.k_pos * np.exp(-self.k_pos_sigma * np.abs(self._pos_abs[0]))

    def _reward_smoothing(self):
        actual_delta_torques = np.sqrt(self.norm(self._get_torque_diff()))
        return self.k_tau * np.exp(-self.k_tau_sigma * actual_delta_torques)

    def _reward_force_contact(self):
        return -self.k_contact * self.get_over_contact_force()

    def _reward_smooth_action(self):
        actual_delta_action = np.sqrt(self.norm(self._get_action_diff())) / 24
        return self.k_action * np.exp(-self.k_action_sigma * actual_delta_action)

    def _reward_config_smooth(self):
        config_des = self.aci._transform_action_to_motor_command(self._old_action)
        actual_config = self._env.robot.GetMotorAngles()
        delta_config = actual_config - config_des
        config_diff = np.sqrt(self.norm(delta_config))
        return self.k_config_smoothing * np.exp(-self.k_config_smoothing_sigma * config_diff)

    def _reward_pitch(self):
        return self.k_pitch * np.exp(-self.k_pitch_sigma * np.abs(self._orient_rpy[1]))

    def _reward_fwd(self):
        return self.k_fwd * self.get_fwd()

    def _reward(self):
        rew = 0
        rew_h = self._reward_height()
        rew_smooth = self._reward_smoothing()
        rew_contact = self._reward_force_contact()
        rew_pitch = self._reward_pitch()
        rew_fwd = self._reward_fwd()

        rew += 0.4 * rew_contact + 0.2 * rew_smooth + 0.25 * rew_h + 0.3 * rew_pitch + 0.4 * rew_fwd

        return rew

    def enable_rest_mode(self):
        self.rest_mode = True

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0

        if not self._terminated():
            reward += 0.05 * (self._max_forward_distance + self._max_height) / 2
        else:
            reward -= 0.0

        return reward


class JumpingInPlacePPOHP(JumpingInPlacePPO):
    def __init__(self, env):
        super().__init__(env)

    def _init_reward_constants(self):
        self.max_height_task = 1.25  # absolute [m]
        self.min_height_task = 0.29  # absolute [m]
        self.max_contact_force = 800  # [N]

    def change_parameters(self):
        if self._env.enable_env_randomization:
            self.max_height_task = 1.1


class JumpingForwardPPOHP(JumpingForwardPPO):
    def __init__(self, env):
        super().__init__(env)

    def _init_reward_constants(self):
        self.max_height_task = 1.1  # absolute [m]
        self.min_height_task = 0.29  # absolute [m]
        self.max_contact_force = 800  # [N]
        self.max_fwd = 1.4

    def change_parameters(self):
        if self._env.enable_env_randomization:
            self.max_height_task = 1.0
            self.max_fwd = 1.3


class BackFlip(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self.maximum_height = 0.7
        self.maximum_pitch = 2 * np.pi
        self.minimum_height = 0.3
        self.max_pitch = 0.0

    def _on_step(self):
        super()._on_step()
        self.max_pitch = max(self.max_pitch, self._orient_rpy[1])

    def _terminated(self):
        return self._is_fallen_ground() or self._not_allowed_contact()

    def _reward_end_episode(self):
        reward = 0
        h_max = np.clip(0, self._max_height - self.minimum_height, self.maximum_height - self.minimum_height) / (
            self.maximum_height - self.minimum_height
        )
        pitch_max = self.max_pitch / self.maximum_pitch

        reward += pitch_max * 0.4
        reward += h_max * 0.4
        reward += h_max * pitch_max

        if self.is_switched_controller():
            if not self._terminated():
                reward += 0.2

        return reward

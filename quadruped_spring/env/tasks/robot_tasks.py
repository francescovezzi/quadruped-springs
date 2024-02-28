import os

import numpy as np

import quadruped_spring as qs
from quadruped_spring.env.sensors.robot_sensors import PitchBackFlip as PBF
from quadruped_spring.env.tasks.task_base import (
    TaskBase,
    TaskContinuousJumping,
    TaskContinuousJumping2,
    TaskJumping,
    TaskJumpingDemo,
    TaskJumpingDemo2,
)


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


class JumpingForwardContinuous(TaskContinuousJumping):
    def __init__(self, env):
        super().__init__(env)
        self.jump_limit = 0.5
        self.time_limit = 0.15

    def _reward(self):
        """Reward for each simulation step."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        # print(self.cumulative_flight_time)
        # print(self.jump_counter)
        max_time_normalized = self.cumulative_flight_time / self.time_limit
        max_distance_normalized = self.cumulative_fwd / self.jump_limit
        bonus_malus_term = (max_time_normalized + max_distance_normalized) / 2

        reward += 0.25 * max_time_normalized
        reward += 0.5 * max_distance_normalized

        reward += max_time_normalized * 0.25 * np.exp(-self._max_pitch**2 / 0.15**2)  # orientation

        if not self._terminated():
            reward += 0.1 * bonus_malus_term
        # else:
        #     reward -= 0.08 * (1 + 1.2 * bonus_malus_term)

        return reward


class JumpingForwardContinuous2(TaskContinuousJumping):
    def __init__(self, env):
        super().__init__(env)
        self.jump_limit = 0.5
        self.time_limit = 0.35

    def _reward(self):
        """Reward for each simulation step."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to be added to the reward at the end of the episode"""
        reward = 0
        # print(self.cumulative_flight_time)
        # print(self.jump_counter)
        max_time_normalized = min(self._max_flight_time, self.time_limit) / self.time_limit
        max_distance_normalized = min(self._max_forward_distance, self.jump_limit) / self.jump_limit
        bonus_malus_term = (max_time_normalized + max_distance_normalized) / 2

        reward += 0.25 * max_time_normalized
        reward += 0.5 * max_distance_normalized

        reward += max_distance_normalized * 0.15 * np.exp(-(self._max_pitch**2 / 0.15**2))  # orientation

        reward += 0.4 * (self._env.get_sim_time() / self._env._MAX_EP_LEN) * bonus_malus_term

        if not self._terminated():
            reward += 0.2 * bonus_malus_term
        # else:
        #     reward -= 0.08 * (1 + 1.2 * bonus_malus_term)

        return reward


class JumpingForwardContinuous3(TaskContinuousJumping2):
    def __init__(self, env):
        super().__init__(env)
        self.jump_limit = 0.6
        self.height_limit = 0.45
        self.fwd_weight = 0.7
        self.height_weight = 0.3
        self.performance_bound = 0.7

    def _reward(self):
        """Reward for each simulation step."""
        return 0

    def _reward_end_episode(self):
        """Compute bonus and malus to be added to the reward at the end of the episode"""

        performance_array = self.get_performance_array()

        while np.size(performance_array) < 3:
            performance_array = np.concatenate((performance_array, np.asarray([0])))

        avg_perf = np.average(performance_array)
        max_perf = np.max(performance_array)
        entropy = self.get_entropy_fwd()
        rew_entropy = np.exp((entropy - 1) / 0.3)
        rew_avg = 0

        rew_avg += avg_perf * 0.15 * np.exp(-self._max_pitch**2 / 0.15**2)  # orientation

        rew_avg += avg_perf * 0.4 * (self._env.get_sim_time() / self._env._MAX_EP_LEN)

        rew_avg += avg_perf * rew_entropy * 0.2

        rew_avg += avg_perf * 0.25

        reward = 0.8 * rew_avg + 0.2 * max_perf

        reward += 0.1 * self.good_jump_counter

        if not self._terminated():
            reward += 0.2 * avg_perf
        # else:
        #     reward -= 0.08 * (1 + 1.2 * bonus_malus_term)

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


class BackflipDemo(TaskJumpingDemo):
    def __init__(self, env):
        self.demo_path = os.path.join(os.path.dirname(qs.__file__), "demonstrations", "backflip-1.npy")
        super().__init__(env)

    def _terminated(self):
        # print(f'{self.demo_counter} / {self.demo_length}')
        return self._is_fallen_ground() or self._not_allowed_contact() or self.demo_length == self.demo_counter


class JumpingContinuousDemoForward(TaskJumpingDemo2):
    def __init__(self, env):
        self.demo_path = os.path.join(os.path.dirname(qs.__file__), "demonstrations", "continuous-jf-1.npy")
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
        self.get_pitch = lambda: PBF._get_pitch(self._env)

    def _on_step(self):
        super()._on_step()
        # print(f"sens: {self.get_pitch()}, rew: {self._orient_rpy[1]}")
        self.max_pitch = max(self.max_pitch, self.get_pitch())

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


class ContinuousJumpingForwardPPO(TaskContinuousJumping2):
    def __init__(self, env):
        super().__init__(env)
        self.aci = self._env.get_ac_interface()
        self._init_reward_constants()
        self._init_reward_parameters()
        self.height_weight = 0.3
        self.fwd_weight = 0.7
        self.jump_limit = 0.6

    def _init_reward_constants(self):
        self.max_height_task = 0.5  # absolute [m]
        self.min_height_task = 0.35  # absolute [m]
        self.max_contact_force = 600  # [N]
        self.max_fwd = 0.9

    @staticmethod
    def norm(vec):
        return np.dot(vec, vec)

    def _init_reward_parameters(self):
        self.k_h = 0.006
        self.k_tau_sigma = 0.15
        self.k_tau = 0.0032
        self.k_contact = 6e-5
        self.k_action_sigma = 0.1
        self.k_action = 0.1
        self.k_config_smoothing_sigma = 5
        self.k_config_smoothing = 0.025
        self.k_pitch_sigma = 26
        self.k_pitch = 0.0043
        self.k_fwd = 0.0075
        self.k_energy = 0.0035
        self.k_energy_sigma = 0.01

    def get_clipped_height(self):
        actual_height = self._pos_abs[2]
        if actual_height < self.min_height_task:
            return 0
        elif actual_height > self.max_height_task:
            return 0
        else:
            return actual_height

    def get_clipped_fwd(self):
        fwd = self.get_actual_fwd()
        return min(fwd, self.jump_limit)

    def get_over_contact_force(self):
        _, _, feet_forces, _ = self._env.robot.GetContactInfo()
        contact_force = np.sum(feet_forces)
        if contact_force > self.max_contact_force:
            # print(f'contact force -> {contact_force}')
            return contact_force - self.max_contact_force
        else:
            return 0

    def get_energy(self):
        dq = self._env.robot.GetMotorVelocities()
        tau = self._env.robot.GetMotorTorques()
        return tau * dq

    def _reset(self):
        super()._reset()

    def _reward_height(self):
        return self.k_h * self.get_clipped_height()

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
        rew = self.k_pitch * np.exp(-self.k_pitch_sigma * np.abs(self._orient_rpy[1]))
        if self.is_jumping:
            return rew * 1.5
        else:
            return rew

    def _reward_fwd(self):
        return self.k_fwd * self.get_actual_fwd()

    def _reward_energy(self):
        energy = np.sqrt(self.norm(self.get_energy()))
        return self.k_energy * np.exp(-self.k_energy_sigma * energy)

    def _reward_end_jump(self):
        rew = 0
        if not self.first_jump and self.end_jump:
            entropy = self.get_entropy_fwd()
            rew_entropy = np.exp((entropy - 1) / 0.3)
            perf_array = self.get_performance_array()
            last_jump_performance = perf_array[-1]
            if last_jump_performance > 0.8:
                rew += last_jump_performance * rew_entropy * 0.35
                rew += last_jump_performance * 0.65
                rew *= 0.2

        return rew

    def _reward(self):
        rew = 0
        if not self.is_switched_controller:
            rew_h = self._reward_height()
            rew_smooth = self._reward_smoothing()
            rew_contact = self._reward_force_contact()
            rew_pitch = self._reward_pitch()
            rew_fwd = self._reward_fwd()
            rew_energy = self._reward_energy()
            rew_end_jump = self._reward_end_jump()
            rew += (
                0.5 * rew_contact
                + 0.2 * rew_smooth
                + 0.3 * rew_h
                + 0.2 * rew_pitch
                + 0.75 * rew_fwd
                + 0.1 * rew_energy
                + 0.2 * rew_end_jump
            ) * 0.8
        return rew

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        entropy = self.get_entropy_fwd()
        rew_entropy = np.exp((entropy - 1) / 0.3)
        avg_perf = self.get_avg_performance()
        reward += avg_perf * rew_entropy
        if not self._terminated():
            return reward
        else:
            return reward - 1


class BackflipPPO(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self.rest_mode = False
        self.aci = self._env.get_ac_interface()
        self._init_reward_constants()
        self._init_reward_parameters()
        self._tot_pitch = 0
        self.max_pitch = 0.0
        self.get_pitch = lambda: PBF._get_pitch(self._env)

    def _init_reward_constants(self):
        self.max_height_task = 0.7  # absolute [m]
        self.min_height_task = 0.29  # absolute [m]
        self.max_contact_force = 800  # [N]
        self.max_fwd = 1.1

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

    def _reset(self):
        super()._reset()

    def _on_step(self):
        super()._on_step()
        self._max_pitch = max(self._max_pitch, self.get_pitch())

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
        if self._pos_abs[2] > 0.5:
            pitch = self.get_pitch()
        else:
            pitch = 0
        return self.k_pitch * pitch

    def _reward(self):
        rew = 0
        rew_h = self._reward_height()
        rew_smooth = self._reward_smoothing()
        rew_contact = self._reward_force_contact()
        rew_pitch = self._reward_pitch()
        self._tot_pitch += rew_pitch
        rew += 0.4 * rew_contact + 0.2 * rew_smooth + 0.25 * rew_h + 0.3 * rew_pitch

        return rew

    def _reward_end_episode(self):
        """Compute bonus and malus to add to reward at the end of the episode"""
        reward = 0
        if not self._terminated():
            reward += 0.2 * (0.7 * self._max_pitch / 5 + 0.3 * self._max_height) / 2
        else:
            reward -= 0.0

        return reward

    def enable_rest_mode(self):
        self.rest_mode = True

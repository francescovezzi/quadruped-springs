import gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped

from quadruped_spring.env.wrappers.get_demonstration_wrapper import GetDemonstrationWrapper


class GoToRestWrapper(gym.Wrapper):
    """Control robot to rest position."""

    def __init__(self, env):
        super().__init__(env)
        self.ac_interface = self.env.get_ac_interface()
        self.des_final_action = self.ac_interface.get_init_action()
        if self.env.are_springs_enabled():
            self.time_interpolation = 1.0
        else:
            self.time_interpolation = 0.3
        self.h_old = 0
        self.h_actual = 0

    def temporary_switch_motor_control_gain(foo):
        def wrapper(self, *args, **kwargs):
            """Temporary switch motor control gain"""
            if self.env.are_springs_enabled():
                kp = 60.0
                # kd = 1.5
                kd = 0.8
            else:
                kp = 60.0
                kd = 1.5
            tmp_save_motor_kp = self.env.robot._motor_model._kp
            tmp_save_motor_kd = self.env.robot._motor_model._kd
            self.env.robot._motor_model._kp = kp
            self.env.robot._motor_model._kd = kd
            ret = foo(self, *args, **kwargs)
            self.env.robot._motor_model._kp = tmp_save_motor_kp
            self.env.robot._motor_model._kd = tmp_save_motor_kd
            return ret

        return wrapper

    def step(self, action):
        # self.step_counter_before_rest += 1
        obs, reward, done, infos = self.env.step(action)
        self.h_old = self.h_actual
        self.h_actual = self.env.robot.GetBasePosition()[2]
        if self.rest_condition() and not done:
            obs, reward, done, infos = self.go_to_rest()
        # if done:
        #     print(f'step counter before going to rest-> {self.step_counter_before_rest}')
        return obs, reward, done, infos

    def get_start_action(self):
        actual_config = self.env.robot.GetMotorAngles()
        return self.ac_interface._transform_motor_command_to_action(actual_config)

    @temporary_switch_motor_control_gain
    def go_to_rest(self):
        if self.env.task_env in [
            "JUMPING_IN_PLACE_PPO",
            "JUMPING_FORWARD_PPO",
            "JUMPING_IN_PLACE_PPO_HP",
            "JUMPING_FORWARD_PPO_HP",
            "BACKFLIP_PPO",
        ]:
            self.env.task.enable_rest_mode()
        done = False
        t_start = self.env.get_sim_time()
        t_end = self.time_interpolation + t_start
        start_action = self.get_start_action()
        if is_wrapped(self.env, GetDemonstrationWrapper):
            self.env.wrapped_demo_env.save_demo()
        while not done:
            action_ref = self.ac_interface.generate_ramp(
                self.env.get_sim_time(), t_start, t_end, start_action, self.des_final_action
            )
            obs, reward, done, infos = self.env.step(action_ref)
        # reward = self.env.get_reward_end_episode()

        return obs, reward, done, infos

    def reset(self):
        obs = self.env.reset()
        # self.step_counter_before_rest = 0
        self.h_old = self.h_actual = self.env.robot.GetBasePosition()[2]
        return obs

    def rest_condition(self):
        _, _, _, feet_in_contact = self.env.robot.GetContactInfo()
        ground_touched = np.all(np.array(feet_in_contact))
        has_jumped = self.env.task.is_switched_controller()
        stop_landing = (self.h_actual - self.h_old) > 0

        return has_jumped and ground_touched and stop_landing

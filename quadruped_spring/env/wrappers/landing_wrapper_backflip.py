import gym
import numpy as np

from quadruped_spring.env.sensors.robot_sensors import PitchBackFlip as PBF
from quadruped_spring.utils.timer import Timer


class LandingWrapperBackflip(gym.Wrapper):
    """
    Wrapper to switch controller when robot starts taking off.
    Dear user please pay attention at the order of the wrapper you are using.
    """

    def __init__(self, env):
        super().__init__(env)
        self._robot_config = self.env.get_robot_config()
        self._landing_action = self.env.get_landing_action()
        # self._take_off_pose = np.array([0, 1.28, -2.57] * 4)
        # self.aci = self.env.get_ac_interface()
        # self.take_off_action = self.aci._transform_motor_command_to_action(self._take_off_pose)
        self.take_off_action = np.array([0, 1, -1, 0, 1, -1])
        self.trigger_pitch = 5 * np.pi / 8
        self.take_off_trigger = lambda: PBF._get_pitch(self.env) >= self.trigger_pitch
        self.timer_jumping = Timer(dt=self.env.env_time_step)

    def temporary_switch_motor_control_gain(foo):
        def wrapper(self, *args, **kwargs):
            """Temporary switch motor control gain"""
            if self.env.are_springs_enabled():
                kp = 60.0
                kd = 1.5
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

    # @temporary_switch_motor_control_gain
    def landing_phase(self):
        action = self._landing_action
        done = False
        while not done:
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def take_off_phase(self):
        """Repeat last action until you rech the height peak"""
        done = False
        while True:
            obs, reward, done, infos = self.env.step(self.take_off_action)
            if self.take_off_trigger() or done:
                break
        return obs, reward, done, infos

    def start_jumping_timer(self):
        actual_time = self.env.get_sim_time()
        delta_time = self.env.task.compute_time_for_peak_heihgt()
        self.timer_jumping.reset_timer()
        self.timer_jumping.start_timer(timer_time=actual_time, start_time=actual_time, delta_time=delta_time)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        if self.env.task.is_switched_controller() and not done:
            _, reward, done, infos = self.take_off_phase()
            if not done:
                _, reward, done, infos = self.landing_phase()

        return obs, reward, done, infos

    def reset(self):
        obs = self.env.reset()
        return obs

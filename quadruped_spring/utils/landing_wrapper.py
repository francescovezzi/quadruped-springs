import gym
import numpy as np

from utils.timer import Timer


class LandingWrapper(gym.Env):
    """ Wrapper to switch controller when robot starts landing"""
    def __init__(self):
        self._landing_pose = self._compute_landing_pose()
        self.timer_jumping = Timer()
    
    def _compute_landing_pose(self):
        landing_pose = np.zeros(self.env._robot_configs.NUM_MOTORS)
        landing_pose = self.env._init_action
        return landing_pose
    
    def temporary_switch_motor_control_gain(foo):
        def wrapper(self, *args, **kwargs):
            """Temporary switch motor control gain"""
            if self.env._enable_springs:
                ret = foo(self, *args, **kwargs)
            else:
                tmp_save_motor_kp = self.env.robot._motor_model._kp
                tmp_save_motor_kd = self.env.robot._motor_model._kd
                self.env.robot._motor_model._kp = 60.0
                self.env.robot._motor_model._kd = 3.0
                ret = foo(self, *args, **kwargs)
                self.env.robot._motor_model._kp = tmp_save_motor_kp
                self.env.robot._motor_model._kd = tmp_save_motor_kd
            return ret
        return wrapper

    @temporary_switch_motor_control_gain
    def landing_phase(self):
        self.env._enable_action_interpolation = True  # Encourage smoothed action when robot is flying
        action = self._landing_pose
        done = False
        while not done:
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos
    
    def take_off_phase(self, action):
        """ Repeat last action until you rech the height peak """
        while not self.timer_jumping.time_up():
            self.start_jumping_timer()
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def is_flying(self):
        return self.env.robot._is_flying()

    def compute_time_for_peak_heihgt(self):
        """Compute the time the robot needs to reach the maximum height
        """
        _, _, vz = self.env.robot.GetBaseVelocity()
        return vz / 9.81
    
    def start_jumping_timer(self):
        actual_time = self.env.get_sim_time()
        self.timer_jumping.start_timer(timer_time=actual_time,
                                        start_time=actual_time,
                                        delta_time=self.compute_time_for_peak_heihgt())        

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        
        if self.is_flying():
            _, reward, done, infos = self.take_off_phase(action)
            _, reward, done, infos = self.landing_phase()
        
        return obs, reward, done, infos
    
    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()
import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper



class EvaluateMetricJumpOnPlace(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.init_metric()
        self.flag_first = True
        
    def compute_max_power(self):
        tau = abs(self.env.robot.GetMotorTorques())
        vel = abs(self.env.robot.GetMotorVelocities())
        
        return max(*(tau * vel)) 
    
    def compute_forward_distance(self):
        x, y, _ = self.env.robot.GetBasePosition()
        dx = x - self.x_pos
        dy = y - self.y_pos
        return np.sqrt(dx**2 + dy**2)

    def init_metric(self):
        self.x_pos, self.y_pos, self.height = self.env.robot.GetBasePosition()
        self.roll, _, self.yaw  = abs(self.env.robot.GetBaseOrientationRollPitchYaw())
        self.power = 0
        self.penalization_invalid_contact = 0

    def eval_metric(self):
        _, numInvalidContacts, _, _ = self.env.robot.GetContactInfo()
        if numInvalidContacts > 0:
            self.penalization_invalid_contact = -10
        self.power = max(self.power, self.compute_max_power())
        roll, _, yaw = self.env.robot.GetBaseOrientationRollPitchYaw()
        _, _, height = self.env.robot.GetBasePosition()
        self.roll = max(self.roll, abs(roll))
        self.yaw = max(self.yaw, abs(yaw))
        self.height = max(self.height, abs(height))

    def get_metric(self):
        rew_dist = 0.2 * np.exp(-self.compute_forward_distance()**2 / 0.1)
        rew_roll =  0.1 * np.exp(-self.roll**2 / 0.1)
        rew_yaw =  0.1 * np.exp(-self.yaw**2 / 0.1)
        max_height = self.height
        max_power = self.power
        
        if abs(max_power) >= 0.01:
            metric = rew_dist + rew_roll + rew_yaw + max_height * 1000 / (2 * max_power)
        else:
            metric = 0
        metric += self.penalization_invalid_contact

        return max(-1, metric)
    
    def print_metric(self):
        print(f"the jump (on place) metric performance amounts to: {self.get_metric()}")
        print(f"the maximum reached height amounts to: {self.height}")

    def step(self, action):

        if self.flag_first:
            self.flag_first = False
            self.init_metric()
        
        obs, reward, done, infos = self.env.step(action)
        self.eval_metric()
        
        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()
        
        
######################################################################

class EvaluateMetricJumpOnPlaceVecEnv(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.env = self.venv
        self.init_metric()
        self.flag_first = True
        
    def compute_max_power(self):
        tau = abs(self.env.robot.GetMotorTorques())
        vel = abs(self.env.robot.GetMotorVelocities())
        
        return max(*(tau * vel)) 
    
    def compute_forward_distance(self):
        x, y, _ = self.env.robot.GetBasePosition()
        dx = x - self.x_pos
        dy = y - self.y_pos
        return np.sqrt(dx**2 + dy**2)

    def init_metric(self):
        self.x_pos, self.y_pos, self.height = self.env.robot.GetBasePosition()
        self.roll, _, self.yaw  = abs(self.env.robot.GetBaseOrientationRollPitchYaw())
        self.power = 0
        self.penalization_invalid_contact = 0

    def eval_metric(self):
        _, numInvalidContacts, _, _ = self.env.robot.GetContactInfo()
        if numInvalidContacts > 0:
            self.penalization_invalid_contact = -10
        self.power = max(self.power, self.compute_max_power())
        roll, _, yaw = self.env.robot.GetBaseOrientationRollPitchYaw()
        _, _, height = self.env.robot.GetBasePosition()
        self.roll = max(self.roll, abs(roll))
        self.yaw = max(self.yaw, abs(yaw))
        self.height = max(self.height, abs(height))

    def get_metric(self):
        rew_dist = 0.2 * np.exp(-self.compute_forward_distance()**2 / 0.1)
        rew_roll =  0.1 * np.exp(-self.roll**2 / 0.1)
        rew_yaw =  0.1 * np.exp(-self.yaw**2 / 0.1)
        max_height = self.height
        max_power = self.power
        
        if abs(max_power) >= 0.01:
            metric = rew_dist + rew_roll + rew_yaw + max_height * 1000 / (2 * max_power)
        else:
            metric = 0
        metric += self.penalization_invalid_contact

        return max(-1, metric)
    
    def print_metric(self):
        print(f"the jump (on place) metric performance amounts to: {self.get_metric()}")
        print(f"the maximum reached height amounts to: {self.height}")

    def step_wait(self):

        if self.flag_first:
            self.flag_first = False
            self.init_metric()
        
        obs, reward, done, infos = self.venv.step_wait()
        self.eval_metric()
        
        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.venv.render(mode, **kwargs)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def close(self):
        self.venv.close()
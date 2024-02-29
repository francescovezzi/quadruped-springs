from typing import Any
import gym
import numpy as np


class EvaluationWrapper(gym.Wrapper):

    def __init__(self, env):
        self.max_h = 0.0
        self.max_fwd = 0.0
        self.time_list = []
        self.height_list = []
        self.x_list = []
        super().__init__(env)
        self.env.set_sub_step_callback(self.fill_lists)
        
    def get_time(self):
        return self.time_list
    
    def get_height(self):
        return self.height_list
    
    def get_x(self):
        return self.x_list
    
    def fill_time(self):
        t = self.env.get_sim_time()
        self.time_list.append(t)
        
    def fill_height_x(self):
        x, _, h = self.env.robot.GetBasePosition()
        self.max_h = max(self.max_h, h)

        self.height_list.append(h)
        self.x_list.append(x)
        
    def fill_lists(self):
        self.max_fwd = max(self.max_fwd, self.env.task.compute_jumping_distance())

        self.fill_time()
        self.fill_height_x()
    
    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
                
        _, _, feet_forces, _ = self.robot.GetContactInfo()
        infos["feet_forces"] = np.sum(feet_forces) / 4
        
        infos["max_height"] = self.max_h
        
        infos["max_fwd"] = self.max_fwd
        
        return obs, reward, done, infos
    
    def reset(self):
        self.max_h = 0.0
        return super().reset()

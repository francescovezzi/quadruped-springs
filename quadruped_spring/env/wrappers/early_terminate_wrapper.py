import gym
import numpy as np


class EarlyTerminateWrapper(gym.Wrapper):
    """
    Wrapper to terminate the episode when the robot starts flying.
    """
    def __init__(self, env):
        super().__init__(env)

    def maximum_height_reached(self):
        height = self.env.robot.GetBasePosition()[2]
        return height >= 0.38

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        if self.maximum_height_reached() and not done:
            reward += self.env.get_reward_end_episode()
            done = True
        
        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        self._init_height = self.env.robot.GetBasePosition()[2]
        return obs

    def close(self):
        self.env.close()
        
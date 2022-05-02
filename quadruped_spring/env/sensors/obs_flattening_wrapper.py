import gym
import numpy as np


class ObsFlatteningWrapper(gym.Wrapper):
    """Wrapper for flattening observation dicitonary. It should be the first wrapper."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation_dict, reward, done, infos = self.env.step(action)

        observations = []
        for key, value in observation_dict.items():
            observations.append(np.asarray(value).flatten())
        flat_observations = np.concatenate(observations)

        return flat_observations, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()

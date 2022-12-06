import gym
import numpy as np


class ObsFlatteningWrapper(gym.Wrapper):
    """Wrapper for flattening observation dicitonary."""

    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def _flatten_obs(observation_dict):
        if isinstance(observation_dict, dict):
            observations = []
            for key, value in observation_dict.items():
                observations.append(np.asarray(value).flatten())
            flat_observations = np.concatenate(observations)
            return flat_observations
        else:
            return observation_dict

    def step(self, action):
        observation_dict, reward, done, infos = self.env.step(action)
        flat_observations = self._flatten_obs(observation_dict)
        return flat_observations, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        observation_dict = self.env.reset()
        flat_observations = self._flatten_obs(observation_dict)
        return flat_observations

    def close(self):
        self.env.close()

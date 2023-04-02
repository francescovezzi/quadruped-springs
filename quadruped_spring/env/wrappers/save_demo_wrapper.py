import gym

from stable_baselines3.common.env_util import is_wrapped
from quadruped_spring.env.wrappers.get_demonstration_wrapper import GetDemonstrationWrapper


class SaveDemoWrapper(gym.Wrapper):
    """Wrapper for saving demonstrations."""

    def __init__(self, env):
        super().__init__(env)
        if not is_wrapped(GetDemonstrationWrapper):
            raise RuntimeError("No Demonstrations to be saved. GetDemonstrationWrapper not used.")

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        if done:
            self.env.wrapped_demo_env.save_demo()
        return obs, reward, done, infos

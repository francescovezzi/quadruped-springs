import gym


class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper to provide curriculum API.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()

    def increase_curriculum_level(self, value):
        """increase the curriculum level."""
        assert value >= 0 and value < 1, "curriculum level change should be in [0,1)."
        self.env.task.increase_curriculum_level(value)

    def get_env_kwargs(self):
        """Get the environment kwargs. Useful for training."""
        action_mode, obs_mode = self.env.get_action_observation_space_mode()
        kwargs = {
            "motor_control_mode": self.env.get_motor_control_mode(),
            "enable_springs": self.env.are_springs_enabled(),
            "enable_action_filter": self.env.low_pass_filter_enabled(),
            "task_env": self.env.task_env,
            "observation_space_mode": obs_mode,
            "action_space_mode": action_mode,
            "curriculum_level": self.env.task.get_curriculum_level(),
        }
        return kwargs

    def print_curriculum_info(self):
        """Print curriculum info."""
        self.env.task.print_curriculum_info()

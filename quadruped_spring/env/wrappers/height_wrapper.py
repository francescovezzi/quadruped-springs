import gym


class HeightWrapper(gym.Wrapper):
    """Wrapper to end episode if robot height is decreasing."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        self._update_height()

        if self._is_height_decreasing():
            done = True

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        self._height_old = self.env.robot.GetBasePosition()[2]
        self._height_new = self._height_old
        return obs

    def _update_height(self):
        self._height_old = self._height_new
        self._height_new = self.env.robot.GetBasePosition()[2]

    def _is_height_decreasing(self):
        return self._height_new - self._height_old < 0

    def close(self):
        self.env.close()

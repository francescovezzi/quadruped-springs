import gym


class HeightWrapper(gym.Wrapper):
    """Wrapper to end episode if robot height is decreasing."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        
        if self.is_flying():
            self._enable_wrapper = False
        
        self._update_height()
        if self._enable_wrapper and self._is_height_decreasing() and not done:
            reward -= 0.005

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        self._enable_wrapper = True
        self._height_init = self.env.robot.GetBasePosition()[2]
        self._height_actual = self._height_init
        return obs

    def _update_height(self):
        self._height_actual = self.env.robot.GetBasePosition()[2]

    def _is_height_decreasing(self):
        return self._height_init - self._height_actual < 0.02

    def is_flying(self):
        return self.env.robot._is_flying() and self.compute_time_for_peak_heihgt() > 0.06

    def compute_time_for_peak_heihgt(self):
        """Compute the time the robot needs to reach the maximum height"""
        _, _, vz = self.env.robot.GetBaseLinearVelocity()
        return vz / 9.81

    def close(self):
        self.env.close()

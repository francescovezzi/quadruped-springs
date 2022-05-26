import gym
import numpy as np

from quadruped_spring.utils.timer import Timer


class RestWrapper(gym.Wrapper):
    """Wrapper to end episode if the robot is at rest for a certain amount of time"""

    def __init__(self, env):
        super().__init__(env)
        self.rest_timer = Timer(dt=self.env.get_env_time_step())
        self._max_rest_time = 0.8

    def _is_rested(self):
        base_lin_vel = self.env.robot.GetBaseLinearVelocity()
        vel_module = np.sqrt(np.dot(base_lin_vel, base_lin_vel))
        return vel_module < 0.001

    def _restart_timer(self):
        actual_time = self.env.get_sim_time()
        self.rest_timer.reset_timer()
        self.rest_timer.start_timer(timer_time=actual_time, start_time=actual_time, delta_time=self._max_rest_time)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        if self._is_rested():
            if not self.rest_timer.already_started():
                self._restart_timer()
            self.rest_timer.step_timer()
        else:
            if self.rest_timer.already_started():
                self.rest_timer.reset_timer()

        if self.rest_timer.time_up():
            # Update sparse reward
            if not done:
                reward += self.env.get_reward_end_episode()
            done = True

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()

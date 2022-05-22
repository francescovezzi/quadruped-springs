import os
import time

import cv2
import gym
import numpy as np


class VideoRec(gym.Wrapper):
    def __init__(self, env: gym.Env, path="logs/videos/", name="rl_video"):
        """Wrapper for video recording.

        Args:
            env (gym.Env): your environment
            path (str, optional): path. Defaults to '/videos'.
            name (str, optional): file name. Defaults to 'rl_video'.
        """
        super().__init__(env)
        self._path = path
        self._name = self._path + name + ".mp4"
        self._init_video()

    def _init_video(self):
        os.makedirs(self._path, exist_ok=True)
        img = self.env.render()
        img_height, img_width, _ = np.shape(img)
        self._time_step = self.env.get_env_time_step()
        self._freq = int(60)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._movie = cv2.VideoWriter(self._name, self._fourcc, self._freq, (img_width, img_height))
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self._movie.write(open_cv_image)

    def _increase_video(self):
        img = self.env.render()
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self._movie.write(open_cv_image)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs

    def release_video(self):
        self._movie.release()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        obs, reward, done, infos = self.env.step(action)

        self._increase_video()
        time.sleep(0.005)
            
        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

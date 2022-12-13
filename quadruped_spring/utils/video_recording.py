import copy
import os

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
        self._old_method = copy.copy(self.unwrapped.step_simulation)
        self.unwrapped.step_simulation = self._step_simulation
        self._path = path
        self._name = os.path.join(self._path, f"{name}.mp4")
        self._init_video()

    def _init_video(self):
        os.makedirs(self._path, exist_ok=True)
        img = self.env.render()
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = np.shape(open_cv_image)
        self._freq = 100
        self._sampling_period = 5
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._movie = cv2.VideoWriter(self._name, self._fourcc, self._freq, (img_width, img_height))
        self.c = 0
        self.stop_recording = False

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
        obs, reward, done, infos = self.env.step(action)
        if done:
            self.release_video()

        return obs, reward, done, infos

    def _step_simulation(self, increase_sim_counter=True):
        self._old_method(increase_sim_counter)
        if increase_sim_counter:
            self.c += 1
            if self.c % self._sampling_period == 0:
                self._increase_video()

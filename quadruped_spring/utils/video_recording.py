import copy
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
        self._old_method = copy.copy(self.unwrapped.step_simulation)
        self.unwrapped.step_simulation = self._step_simulation
        self._path = path
        self._name = self._path + name + ".mp4"
        self._n_episodes = 1
        self._episode_counter = 0
        self._disable_video = False
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
        self._video_counter = 0
        self._episode_counter += 1
        if self._episode_counter <= self._n_episodes:
            obs = self.env.reset()
        else:
            obs = self.env.get_observation()
        return obs

    def release_video(self):
        self._movie.release()

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def _step_simulation(self, increase_sim_counter=True):
        self._old_method(increase_sim_counter)
        self._video_counter += 1
        if self._video_counter == 10:
            self._increase_video()
            self._video_counter = 0
        time.sleep(0.005)

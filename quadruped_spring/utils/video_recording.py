import os
import time

import cv2
import gym
import numpy as np


class VideoRec(gym.Wrapper):
    def __init__(self, env: gym.Env, video_length, path="logs/videos/", name="rl_video", release=True):
        # Call the parent constructor, so we can access self.env later
        """_summary_

        Args:
            env (gym.Env): your environment
            path (str, optional): path. Defaults to '/videos'.
            name (str, optional): file name. Defaults to 'rl_video'.
            video_length (int): number of simulation steps to record.
            release (bool, optional): Video released autmatically. Defaults to True.
        """
        super().__init__(env)
        self._path = path
        self._name = self._path + name + ".mp4"
        self._video_length = video_length
        self._step_counter = 0
        self._release = release
        self._video_done = False
        self._init_video()

    def _init_video(self):
        os.makedirs(self._path, exist_ok=True)
        img = self.env.render()
        img_height, img_width, _ = np.shape(img)
        self._time_step = self.env._time_step * self.env._action_repeat
        self._freq = int(1 / self._time_step)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._movie = cv2.VideoWriter(self._name, self._fourcc, self._freq, (img_width, img_height))
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self._movie.write(open_cv_image)
        self._step_counter += 1

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

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        obs, reward, done, infos = self.env.step(action)
        self._step_counter += 1

        if not self._video_done:
            if self._step_counter < self._video_length:
                self._increase_video()
                time.sleep(0.005)
            elif self._step_counter == self._video_length or done:
                if self._release:
                    self._movie.release()
                    self._video_done = True
                    time.sleep(0.5)
            else:
                pass

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

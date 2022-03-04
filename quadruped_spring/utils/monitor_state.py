import gym
import matplotlib
import numpy as np

matplotlib.use("tkagg")
import os
import time

import matplotlib.pyplot as plt

NUM_MOTORS = 12
TIME_STEP = 0.001


class MonitorState(gym.Wrapper):
    def __init__(self, env: gym.Env, path="logs/plots/", rec_length=1000, paddle=10, release=True):
        super().__init__(env)
        self._path = path
        self._paddle = paddle
        self._rec_length = rec_length
        self._time_step = TIME_STEP
        self._init_storage()
        self.quadruped = self.env.robot
        self._motor = self.env.robot._motor_model
        self._spring_stiffness = self._motor.getSpringStiffness()
        self._spring_rest_angles = self._motor.getSpringRestAngles()
        self._step_counter = 0
        self._plot_counter = 0
        self._release = release
        self._plot_done = False
        # self._check()

    def _check(self):  # cannot access to private attribute _is_render
        if self.env._is_render:
            raise ValueError("I think it's better if you disable rendering for wrapper MonitorState")

    def _init_storage(self):
        self._length = self._rec_length // self._paddle
        self._time = np.zeros(self._length)
        self._energy_spring = np.zeros((self._length, NUM_MOTORS))
        self._tau_spring = np.zeros((self._length, NUM_MOTORS))
        self._config = np.zeros((self._length, NUM_MOTORS))
        self._motor_vel = np.zeros((self._length, NUM_MOTORS))
        self._motor_tau = np.zeros((self._length, NUM_MOTORS))
        self._base_pos = np.zeros((self._length, 3))
        self._base_or = np.zeros((self._length, 3))

    def _compute_energy_spring(self, q):
        q_bar = self._spring_rest_angles
        k = self._spring_stiffness
        U = 0.5 * k * (q - q_bar) ** 2
        return U

    def _get_data(self, i):
        self._time[i] = self._step_counter * self._time_step
        self._config[i, :] = self.quadruped.GetMotorAngles()
        self._motor_vel[i, :] = self.quadruped.GetMotorVelocities()
        self._motor_tau[i, :] = self.quadruped._applied_motor_torque
        self._tau_spring[i, :] = self.quadruped._spring_torque
        self._energy_spring[i, :] = self._compute_energy_spring(self._config[i, :])
        self._base_pos[i, :] = self.quadruped.GetBasePosition()
        self._base_or[i, :] = self.quadruped.GetBaseOrientationRollPitchYaw()

    def _plot12(self, state, title, ylabel):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL")
        fig.suptitle(title)
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = state[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel(ylabel, rotation=0)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # plt.show()
        return fig, axs

    def _plot3(self, rpy, title, ylabels):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        fig.suptitle(title)
        for i, (ax, ylab) in enumerate(zip(axs, ylabels)):
            data = rpy[:, i]
            ax.plot(self._time, data)
            ax.set_xlabel("t")
            ax.set_ylabel(ylab)
        return fig, axs

    def _create_plots(self):
        fig_config, _ = self._plot12(self._config, "configuration", ylabel="$q$")
        fig_rpy, _ = self._plot3(self._base_or, "Base Orientation", ["roll", "pitch", "yaw"])
        fig_pos, _ = self._plot3(self._base_pos, "Base Position", ["x", "y", "x"])
        fig_motor_vel, _ = self._plot12(self._motor_vel, "Motor velocities", "$\\omega$")
        fig_motor_tau, _ = self._plot12(self._motor_tau, "Motor taus", "$\\tau$")
        fig_tau_spring, _ = self._plot12(self._tau_spring, "Spring taus", "$\\tau$")
        fig_energy_spring, _ = self._plot12(self._energy_spring, "Spring Energy", "$u$")
        figs = [fig_config, fig_rpy, fig_pos, fig_motor_vel, fig_motor_tau, fig_tau_spring, fig_energy_spring]
        names = ["config", "orientation", "position", "motor_vel", "motor_tau", "spring_tau", "spring_energy"]
        return dict(zip(figs, names))

    def _store_plots(self):
        os.makedirs(self._path, exist_ok=True)
        dict = self._create_plots()
        for fig, name in dict.items():
            fig.savefig(os.path.join(self._path, name))

    def step(self, action):

        obs, reward, done, infos = self.env.step(action)

        if not self._plot_done:
            if self._step_counter % self._paddle == 0:
                self._get_data(self._plot_counter)
                time.sleep(0.05)
                self._plot_counter += 1
            self._step_counter += 1

            if self._step_counter == self._rec_length and self._release:
                self._store_plots()
                self._plot_done = True
                time.sleep(1)

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

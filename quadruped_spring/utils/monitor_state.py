import gym
import matplotlib
import numpy as np

matplotlib.use("tkagg")
import os
import time

import matplotlib.pyplot as plt


class MonitorState(gym.Wrapper):
    def __init__(self, env: gym.Env, path="logs/plots/", rec_length=1000, paddle=10, release=True):
        super().__init__(env)
        self._path = path
        self._paddle = paddle
        self._rec_length = rec_length
        self._time_step = self.env._time_step
        self._init_storage()
        self.quadruped = self.env.robot
        self._motor = self.env.robot._motor_model
        self._spring_stiffness = self._motor.getSpringStiffness()
        self._spring_rest_angles = self._motor.getSpringRestAngles()
        self._step_counter = 0
        self._plot_counter = 0
        self._release = release
        self._plot_done = False
        self._h_min = 0.15

    def _init_storage(self):
        NUM_MOTORS = self.env._robot_config.NUM_MOTORS
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
        if self.env._enable_springs:
            q_bar = self._spring_rest_angles
            k = self._spring_stiffness
            U = 0.5 * k * (q - q_bar) ** 2
            return U
        else:
            return np.zeros(12)

    def _get_data(self, i):
        self._time[i] = self.env.get_sim_time()
        self._config[i, :] = self.quadruped.GetMotorAngles()
        self._motor_vel[i, :] = self.quadruped.GetMotorVelocities()
        self._motor_tau[i, :] = self.env.robot._applied_motor_torque
        self._tau_spring[i, :] = self.env.robot._spring_torque
        self._energy_spring[i, :] = self._compute_energy_spring(self._config[i, :])
        self._base_pos[i, :] = self.quadruped.GetBasePosition()
        self._base_or[i, :] = self.quadruped.GetBaseOrientationRollPitchYaw()

    def _plot12(self, state, title, ylabel, limits=([False] * 3, [None] * 3)):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL", "up_limit", "low_limit")
        fig.suptitle(title)
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = state[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel(ylabel, rotation=0)
            if limits[0][i]:
                length = np.shape(self._time)[0]
                n_lines = np.shape(limits[1][i])[0]
                limits_values = np.zeros((length, n_lines))
                for line in range(n_lines):
                    limits_values[:, line] = np.full(length, limits[1][i][line])
                ax.plot(self._time, limits_values, "--")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # plt.show()
        return fig, axs

    def _plot3(self, rpy, title, ylabels, limits=([False] * 3, [None] * 3)):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        fig.suptitle(title)
        for i, (ax, ylab) in enumerate(zip(axs, ylabels)):
            data = rpy[:, i]
            ax.plot(self._time, data)
            ax.set_xlabel("t")
            ax.set_ylabel(ylab)
            if limits[0][i]:
                length = np.shape(self._time)[0]
                n_lines = np.shape(limits[1])[0]
                limits_values = np.zeros((length, n_lines))
                for line in range(n_lines):
                    limits_values[:, line] = np.full(length, limits[1][line])
                ax.plot(self._time, limits_values, "--")
                labels = [ylabels[i], "limit"]
                ax.legend(labels, loc="best")
        return fig, axs

    def _create_plots(self):
        tau_lim = self.env._robot_config.TORQUE_LIMITS[0:3]
        print(tau_lim)
        tau_lim_aux = np.stack([tau_lim, -tau_lim], axis=1)
        tau_limits = ([True] * 3, tau_lim_aux)
        
        pos_limits = ([False, False, True], [None, None, self._h_min])

        joint_lim_up = self.env._robot_config.RL_UPPER_ANGLE_JOINT[0:3]
        joint_lim_down = self.env._robot_config.RL_LOWER_ANGLE_JOINT[0:3]
        joint_lim_aux = np.stack([joint_lim_up, joint_lim_down], axis=1)
        joint_limits = ([True] * 3, joint_lim_aux)
        
        velocity_lim = self.env._robot_config.VELOCITY_LIMITS[0:3]
        velocity_lim_aux = np.stack([velocity_lim, -velocity_lim], axis=1)
        velocity_limits = ([True] * 3, velocity_lim_aux)
        fig_config, _ = self._plot12(self._config, "configuration", "$q$", joint_limits)
        fig_rpy, _ = self._plot3(self._base_or, "Base Orientation", ["roll", "pitch", "yaw"])
        fig_pos, _ = self._plot3(self._base_pos, "Base Position", ["x", "y", "z"], pos_limits)
        fig_motor_vel, _ = self._plot12(self._motor_vel, "Motor velocities", "$\\omega$", velocity_limits)
        fig_motor_tau, _ = self._plot12(self._motor_tau, "Motor taus", "$\\tau$", tau_limits)
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
        self._plot_done = True

    def step(self, action):
        
        obs, reward, done, infos = self.env.step(action)
        self._step_counter += 1

        if not self._plot_done:
            if self._step_counter % self._paddle == 0:
                self._get_data(self._plot_counter)
                time.sleep(0.05)
                self._plot_counter += 1
            if self._release:
                if self._step_counter == self._rec_length or done:
                    self._store_plots()
                    time.sleep(1)

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

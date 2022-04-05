import gym
import matplotlib
import numpy as np

matplotlib.use("tkagg")
import os
import time

import matplotlib.pyplot as plt

H_MIN = 0.15


class MonitorState(gym.Wrapper):
    def __init__(self, env, path="logs/plots/", rec_length=1000, paddle=10):
        super().__init__(env)
        self._path = path
        self._paddle = paddle
        self._rec_length = rec_length
        self._time_step = self.env._time_step
        self.quadruped = self.env.robot
        self._motor = self.env.robot._motor_model
        self._spring_stiffness = np.array(self.env._robot_config.SPRINGS_STIFFNESS * 4)
        self._spring_rest_angles = np.array(self.env._robot_config.SPRINGS_DAMPING * 4)
        self._step_counter = 0
        self._plot_counter = 0
        self._torque_limits = self.env._robot_config.TORQUE_LIMITS
        self._velocity_limits = self.env._robot_config.VELOCITY_LIMITS

        self._init_storage()

    def _init_storage(self):
        NUM_MOTORS = self.env._robot_config.NUM_MOTORS
        self._length = self._rec_length // self._paddle
        self._time = np.zeros(self._length)
        self._energy_spring = np.zeros((self._length, NUM_MOTORS))
        self._tau_spring = np.zeros((self._length, NUM_MOTORS))
        self._config = np.zeros((self._length, NUM_MOTORS))
        self._motor_true_vel = np.zeros((self._length, NUM_MOTORS))
        self._motor_estimate_vel = np.zeros((self._length, NUM_MOTORS))
        self._motor_tau = np.zeros((self._length, NUM_MOTORS))
        self._base_pos = np.zeros((self._length, 3))
        self._base_or = np.zeros((self._length, 3))
        self._feet_normal_forces = np.zeros((self._length, 4))

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
        self._motor_true_vel[i, :] = self.quadruped.GetMotorVelocities()
        self._motor_estimate_vel[i, :] = self.env.get_joint_velocity_estimation()
        self._motor_tau[i, :] = self.env.robot._applied_motor_torque
        self._tau_spring[i, :] = self.env.robot._spring_torque
        self._energy_spring[i, :] = self._compute_energy_spring(self._config[i, :])
        self._base_pos[i, :] = self.quadruped.GetBasePosition()
        self._base_or[i, :] = self.quadruped.GetBaseOrientationRollPitchYaw()
        self._feet_normal_forces[i, :] = self.quadruped.GetContactInfo()[2]

    def _plot_normal_forces(self):
        fig, ax = plt.subplots()
        labels = ["RR", "RL", "FR", "FL"]
        fig.suptitle("feet normal forces")
        ax.plot(self._time, self._feet_normal_forces)
        ax.set_xlabel("t")
        ax.set_ylabel("F", rotation=0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        return fig, ax

    def _plot_height(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        height = self._base_pos[:, 2]
        ax.plot(self._time, height)
        ax.set_title("height(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("h", rotation=0)
        length = np.shape(self._time)[0]
        limit = np.full(length, H_MIN)
        ax.plot(self._time, limit, "--")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(["h", "h_min"], loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        return fig, ax

    def _plot_motor_torques(self):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL", "up_limit", "low_limit")
        fig.suptitle("motor torques")
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = self._motor_tau[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel("tau", rotation=0)
            length = np.shape(self._time)[0]
            limit_up = self._torque_limits[i]
            limit_low = -limit_up
            ax.plot(self._time, np.full(length, limit_up), "--")
            ax.plot(self._time, np.full(length, limit_low), "--")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # plt.show()
        return fig, axs

    def _plot_true_motor_velocities(self):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL")  #  , "up_limit", "low_limit")
        fig.suptitle("motor true velocities")
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = self._motor_true_vel[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel("w", rotation=0)
            # length = np.shape(self._time)[0]
            # limit_up = self._velocity_limits[i]
            # limit_low = - limit_up
            # ax.plot(self._time, np.full(length, limit_up), "--")
            # ax.plot(self._time, np.full(length, limit_low), "--")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # plt.show()
        return fig, axs

    def _plot_estimate_motor_velocities(self):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL")
        fig.suptitle("motor  velocities estimation")
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = self._motor_estimate_vel[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel("w hat", rotation=0)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # plt.show()
        return fig, axs

    def _generate_figs(self):
        fig_height, _ = self._plot_height()
        fig_motor_torque, _ = self._plot_motor_torques()
        fig_motor_true_velocity, _ = self._plot_true_motor_velocities()
        fig_motor_hat_velocity, _ = self._plot_estimate_motor_velocities()
        fig_feet_normal_forces, _ = self._plot_normal_forces()

        figs = [fig_height, fig_motor_torque, fig_motor_true_velocity, fig_motor_hat_velocity, fig_feet_normal_forces]
        names = ["height", "motor_torque", "motor_true_velocity", "motor_hat_velocity", "feer_normal_forces"]
        return dict(zip(figs, names))

    def release_plots(self):
        os.makedirs(self._path, exist_ok=True)
        dict = self._generate_figs()
        for fig, name in dict.items():
            fig.savefig(os.path.join(self._path, name))
        self._plot_done = True

    def step(self, action):

        obs, reward, done, infos = self.env.step(action)
        self._step_counter += 1

        if self._step_counter % self._paddle == 0 and self._plot_counter < self._length:
            self._get_data(self._plot_counter)
            self._plot_counter += 1

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

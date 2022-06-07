import gym
import matplotlib
import numpy as np

matplotlib.use("tkagg")
import copy
import os
import time

import matplotlib.pyplot as plt

H_MIN = 0.15


class MonitorState(gym.Wrapper):
    def __init__(self, env, path="logs/plots/", paddle=10):
        super().__init__(env)
        self._old_method = copy.copy(self.unwrapped.step_simulation)
        self.unwrapped.step_simulation = self._step_simulation
        self._path = path
        self._paddle = paddle
        # self._rec_length = rec_length
        self._time_step = 0.001
        self.quadruped = self.env.robot
        self._motor = self.quadruped._motor_model
        if self.quadruped._enable_springs:
            self._spring_rest_angles = np.array(
                self.quadruped._robot_config.SPRINGS_REST_ANGLE * self.quadruped._robot_config.NUM_LEGS
            )
        self._step_counter = 0
        self._torque_limits = self.quadruped._robot_config.TORQUE_LIMITS
        self._velocity_limits = self.quadruped._robot_config.VELOCITY_LIMITS
        self._plot_done = False
        self._n_episodes = 1
        self._episode_counter = 0

    # def _init_storage(self):
    #     NUM_MOTORS = self.env._robot_config.NUM_MOTORS
    #     self._length = self._rec_length // self._paddle
    #     self._time = np.zeros(self._length)
    #     self._energy_spring = np.zeros((self._length, NUM_MOTORS))
    #     self._tau_spring = np.zeros((self._length, NUM_MOTORS))
    #     self._config = np.zeros((self._length, NUM_MOTORS))
    #     self._motor_true_vel = np.zeros((self._length, NUM_MOTORS))
    #     self._motor_estimate_vel = np.zeros((self._length, NUM_MOTORS))
    #     self._motor_tau = np.zeros((self._length, NUM_MOTORS))
    #     self._base_pos = np.zeros((self._length, 3))
    #     self._base_or = np.zeros((self._length, 3))
    #     self._feet_normal_forces = np.zeros((self._length, 4))

    def _init_storage(self):
        self._length = []
        self._time = []
        self._energy_spring = []
        self._tau_spring = []
        self._config = []
        self._motor_true_vel = []
        self._motor_tau = []
        self._base_pos = []
        self._base_or = []
        self._feet_normal_forces = []
        self._pitch = []
        self._pitch_rate = []

    def _compute_energy_spring(self, q):
        if self.quadruped._enable_springs:
            q_bar = self._spring_rest_angles
            k, _, _ = self.env.robot.get_spring_real_stiffness_and_damping()
            U = 0.5 * k * (q - q_bar) ** 2
            return U
        else:
            return np.zeros(12)

    def _get_data(self):
        self._time.append(self._step_counter * self._time_step)
        self._config.append(self.quadruped.GetMotorAngles())
        self._motor_true_vel.append(self.quadruped.GetMotorVelocities())
        self._motor_tau.append(self.env.robot._applied_motor_torque)
        self._tau_spring.append(self.env.robot._spring_torque)
        self._energy_spring.append(self._compute_energy_spring(self._config[-1]))
        self._base_pos.append(self.quadruped.GetBasePosition())
        self._base_or.append(self.quadruped.GetBaseOrientationRollPitchYaw())
        self._feet_normal_forces.append(self.quadruped.GetContactInfo()[2])
        self._pitch_rate.append([self.quadruped.GetTrueBaseRollPitchYawRate()[1]])

    def _transform_to_array(self):
        self._time = np.asarray(self._time)
        self._config = np.asarray(self._config)
        self._motor_true_vel = np.asarray(self._motor_true_vel)
        self._motor_tau = np.asarray(self._motor_tau)
        self._tau_spring = np.asarray(self._tau_spring)
        self._energy_spring = np.asarray(self._energy_spring)
        self._base_pos = np.asarray(self._base_pos)
        self._base_or = np.asarray(self._base_or)
        self._feet_normal_forces = np.asarray(self._feet_normal_forces)
        self._pitch_rate = np.asarray(self._pitch_rate)

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
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(["h", "h_min"], loc="center left", bbox_to_anchor=(1, 0.5))
        # plt.tight_layout(rect=[0, 0, 0.75, 1])
        ax.legend(["h", "h_min"])
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

    def _plot_config(self):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL")  #  , "up_limit", "low_limit")
        fig.suptitle("motor angles")
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = self._config[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel("q", rotation=0)
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

    def _plot_true_motor_velocities(self):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL")  #  , "up_limit", "low_limit")
        fig.suptitle("motor velocities")
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

    def _plot_springs(self):
        fig, axs = plt.subplots(nrows=3, sharex=True)
        titles = ["HIP", "THIGH", "CALF"]
        labels = ("FR", "FL", "RR", "RL")
        fig.suptitle("elastic energy")
        for i, (ax, title) in enumerate(zip(axs, titles)):
            data = self._energy_spring[:, i + np.array([0, 3, 6, 9])]
            ax.plot(self._time, data)
            ax.set_title(title)
            ax.set_xlabel("t")
            ax.set_ylabel("U", rotation=0)
            length = np.shape(self._time)[0]
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # plt.show()
        return fig, axs

    def _plot_jump(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        title = "Jump forward motion"
        x_pos, z_pos = self._base_pos[:, 0], self._base_pos[:, 2]
        ax.plot(x_pos, z_pos)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("h", rotation=0)
        # plt.show()
        return fig, ax

    def _plot_pitch_rate(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        pitch_rate = self._pitch_rate
        ax.plot(self._time, pitch_rate)
        ax.set_title("pitch rate")
        ax.set_xlabel("t")
        ax.set_ylabel(r"\dot{p}", rotation=0)
        return fig, ax

    def _plot_pitch(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        pitch = self._base_or[:, 1]
        ax.plot(self._time, pitch)
        ax.set_title("pitch")
        ax.set_xlabel("t")
        ax.set_ylabel(r"p", rotation=0)
        return fig, ax

    def _generate_figs(self):
        self._transform_to_array()
        fig_height, _ = self._plot_height()
        fig_motor_torque, _ = self._plot_motor_torques()
        fig_motor_true_velocity, _ = self._plot_true_motor_velocities()
        fig_feet_normal_forces, _ = self._plot_normal_forces()
        fig_springs, _ = self._plot_springs()
        fig_jump, _ = self._plot_jump()
        fig_pitch, _ = self._plot_pitch()
        fig_pitch_rate, _ = self._plot_pitch_rate()
        fig_config, _ = self._plot_config()

        figs = [
            fig_height,
            fig_config,
            fig_motor_torque,
            fig_motor_true_velocity,
            fig_feet_normal_forces,
            fig_springs,
            fig_jump,
            fig_pitch,
            fig_pitch_rate,
        ]
        names = [
            "height",
            "angles",
            "motor_torque",
            "motor_true_velocity",
            "feer_normal_forces",
            "elastic_potential_energy",
            "forward_jumping",
            "pitch",
            "pitch_rate",
        ]
        return dict(zip(figs, names))

    def release_plots(self):
        os.makedirs(self._path, exist_ok=True)
        dict = self._generate_figs()
        for fig, name in dict.items():
            fig.savefig(os.path.join(self._path, name))

    def step(self, action):

        obs, reward, done, infos = self.env.step(action)

        return obs, reward, done, infos

    def reset(self):
        self._episode_counter += 1
        if self._episode_counter <= self._n_episodes:
            self._init_storage()
            obs = self.env.reset()
        else:
            obs = self.env.get_observation()
        return obs

    def _step_simulation(self, increase_sim_counter=True):
        self._old_method(increase_sim_counter)
        self._step_counter += 1

        if self._episode_counter <= self._n_episodes and self._step_counter % self._paddle == 0:
            self._get_data()

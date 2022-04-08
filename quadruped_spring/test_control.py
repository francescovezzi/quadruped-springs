import time

import numpy as np

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.monitor_state import MonitorState


class StateMachine(QuadrupedGymEnv):
    def __init__(self, on_rack=False, render=True, enable_springs=True, enable_joint_velocity_estimate=False, motor_control_mode = 'CARTESIAN_PD'):
        super().__init__(
            on_rack=on_rack,
            render=render,
            enable_springs=enable_springs,
            enable_joint_velocity_estimate=enable_joint_velocity_estimate,
            motor_control_mode=motor_control_mode,
        )

    def temporary_switch_motor_control_mode(mode):
        def _decorator(foo):
            def wrapper(self, *args, **kwargs):
                """Settle robot and add noise to init configuration."""
                # change to 'mode' control mode to set initial position, then set back..
                tmp_save_motor_control_mode_ENV = self._motor_control_mode
                tmp_save_motor_control_mode_ROB = self.robot._motor_control_mode
                self._motor_control_mode = mode
                self.robot._motor_control_mode = mode
                try:
                    tmp_save_motor_control_mode_MOT = self.robot._motor_model._motor_control_mode
                    self.robot._motor_model._motor_control_mode = mode
                except:
                    pass
                foo(self, *args, **kwargs)
                # set control mode back
                self._motor_control_mode = tmp_save_motor_control_mode_ENV
                self.robot._motor_control_mode = tmp_save_motor_control_mode_ROB
                try:
                    self.robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
                except:
                    pass

            return wrapper

        return _decorator

    # @temporary_switch_motor_control_mode(mode="PD")
    # def settle_init_config(self):
    #     init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
    #     for _ in range(800):
    #         self.robot.ApplyAction(init_motor_angles)
    #         self._pybullet_client.stepSimulation()
    #         if self._is_render:
    #             self._render_step_helper()
    #         print(self.robot.GetBasePosition()[2])

    # @temporary_switch_motor_control_mode(mode="TORQUE")
    # def settle_init_config2(self):
    #     action_ref = np.full(12,0)
    #     action_ref[2] = 1
    #     for _ in range(8000):
    #         command = self.ScaleActionToCartesianPos(action_ref)
    #         self.robot.ApplyAction(command)
    #         self._pybullet_client.stepSimulation()

    def _print_err_config(self, config_des):
        q = self.robot.GetMotorAngles()
        err = q - config_des
        print(f'configuration error: {err}')
        
    def _print_height(self):
        print(f'robot base height = {self.robot.GetBasePosition()[2]}')

    # @temporary_switch_motor_control_mode(mode="TORQUE")
    def test_joint_control_cartesian_mode(self, sim_steps):
        # self.robot._motor_model._kp = np.array([600, 600, 600] * 4)
        # self.robot._motor_model._kd = np.array([1.2, 1.2, 1.2] * 4)
        config_des = self.height_to_theta_des(0.15)
        action = np.array([0,0,1]*4)
        for _ in range(sim_steps):
            env.step(action)
            self._print_err_config(config_des)
            self._print_height()
            if self._is_render:
                self._render_step_helper()

    @temporary_switch_motor_control_mode(mode="PD")
    def test_joint_control_torque_mode(self, sim_steps):
        # self.robot._motor_model._kp = np.array([600, 600, 600] * 4)
        # self.robot._motor_model._kd = np.array([1.2, 1.2, 1.2] * 4)
        config_des = self.height_to_theta_des(0.15)
        action = self._map_command_to_action(config_des)
        for _ in range(sim_steps):
            env.step(action)
            self._print_err_config(config_des)
            if self._is_render:
                self._render_step_helper()

    @temporary_switch_motor_control_mode(mode="TORQUE")
    def settle_init_config(self, sim_steps):
        config_des = self._robot_config.INIT_MOTOR_ANGLES
        # config_des = self.height_to_theta_des(0.16)
        for _ in range(sim_steps):
            command = self.angle_ref_to_command(config_des)
            self.robot.ApplyAction(command)
            self._pybullet_client.stepSimulation()
            if self._is_render:
                self._render_step_helper()
            # print(self.robot.GetBasePosition()[2])

    @temporary_switch_motor_control_mode(mode="TORQUE")
    def jump(self):
        coeff = 0.35
        f_rear = 180
        f_front = coeff * f_rear
        jump_command = np.full(12, 0)
        for i in range(4):
            if i < 2:
                f = f_front
            else:
                f = f_rear
            jump_command[3 * i : 3 * (i + 1)] = self.map_force_to_tau([0, 0, -f], i)
            jump_command[3 * i] = 0
        print(jump_command)
        for _ in range(1500):
            self.robot.ApplyAction(jump_command)
            self._pybullet_client.stepSimulation()
            if self._is_render:
                self._render_step_helper()

    def map_force_to_tau(self, F_foot, i):
        J, _ = self.robot.ComputeJacobianAndPosition(i)
        tau = J.T @ F_foot
        return tau

    def generate_ramp(self, i, i_min, i_max, u_min, u_max) -> float:
        if i < i_min:
            return u_min
        elif i > i_max:
            return u_max
        else:
            return u_min + (u_max - u_min) * (i - i_min) / (i_max - i_min)

    def height_to_theta_des(self, h):
        l = self._robot_config.THIGH_LINK_LENGTH
        theta_thigh = np.arccos(h / (2 * l))
        theta_des = np.array([0, theta_thigh, -2 * theta_thigh] * 4)
        return theta_des

    def angle_ref_to_command(self, angles_ref):
        q = self.robot.GetMotorAngles()
        dq = self.robot.GetMotorVelocities()
        if self._enable_springs:
            kp = 200
            kd = 1.0
        else:
            kp = 55
            kd = 0.8
        torque = -kp * (q - angles_ref) - kd * dq
        print(f'angle error: {q - angles_ref}')
        return torque

    @temporary_switch_motor_control_mode(mode="TORQUE")
    def couch(self, sim_steps):
        assert sim_steps > 1000, "simulation time > 1000 for this phase"
        i_min = 0
        i_max = sim_steps - 500
        config_init = self.robot.GetMotorAngles()
        config_des = self.height_to_theta_des(0.16)
        for i in range(sim_steps):
            config_ref = [self.generate_ramp(i, i_min, i_max, config_init[j], config_des[j]) for j in range(12)]
            command = self.angle_ref_to_command(config_ref)
            self.robot.ApplyAction(command)
            self._pybullet_client.stepSimulation()
            if self._is_render:
                self._render_step_helper()


def build_env():
    env_config = {}
    env_config["enable_springs"] = True
    env_config["render"] = True
    env_config["on_rack"] = False
    env_config["enable_joint_velocity_estimate"] = False
    env_config["motor_control_mode"] = "CARTESIAN_PD"

    return StateMachine(**env_config)


if __name__ == "__main__":

    env = build_env()
    sim_steps = 1000
    env = MonitorState(env, path="logs/plots/test_control", rec_length=sim_steps)

    env.test_joint_control_cartesian_mode(sim_steps)
    # env.test_joint_control_torque_mode(sim_steps)
    
    env.close()
    print("end")

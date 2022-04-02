import time

import numpy as np

from env.quadruped_gym_env import QuadrupedGymEnv


class StateMachine(QuadrupedGymEnv):
    def __init__(self, on_rack=False, render=True, enable_springs=True, enable_joint_velocity_estimate=False):
        super().__init__(
            on_rack=on_rack,
            render=render,
            enable_springs=enable_springs,
            enable_joint_velocity_estimate=enable_joint_velocity_estimate,
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

    @temporary_switch_motor_control_mode(mode="PD")
    def settle_init_config(self):
        init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
        for _ in range(800):
            self.robot.ApplyAction(init_motor_angles)
            self._pybullet_client.stepSimulation()
            if self._is_render:
                self._render_step_helper()
                
    @temporary_switch_motor_control_mode(mode="TORQUE")
    def settle_init_config2(self):
        action_ref = np.full(12,0)
        action_ref[2] = 1
        for _ in range(8000):
            command = self.ScaleActionToCartesianPos(action_ref)
            self.robot.ApplyAction(command)
            self._pybullet_client.stepSimulation()


    def cartesian_(self, F_foot):
        taus = np.full(12, 0)
        for i in range(4):
            J, xyz = self.robot.ComputeJacobianAndPosition(i)
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau = J.T @ F_foot
            taus[3 * i : 3 * (i + 1)] = tau  # TODO: add white noise
        return taus

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
            kd = 1.5
        else:
            kp = 100
            kd = 1
        torque = -kp * (q - angles_ref) - kd * dq
        return np.clip(torque, -20, 20)

    @temporary_switch_motor_control_mode(mode="TORQUE")
    def couch(self):
        i_min = 0
        i_max = 4000
        config_init = self._robot_config.INIT_MOTOR_ANGLES
        config_des = self.height_to_theta_des(0.06)
        for i in range(i_max + 1000):
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
    env_config["on_rack"] = True
    env_config["enable_joint_velocity_estimate"] = False

    return StateMachine(**env_config)


if __name__ == "__main__":

    env = build_env()
    sim_steps = 1000

    env.settle_init_config2()
    # env.couch()

    # obs = env.reset()
    # for i in range(sim_steps):
    #     action = np.random.rand(12) * 2 - 1
    #     # action = np.full(12,0)
    #     obs, reward, done, info = env.step(action)
    env.close()
    print("end")

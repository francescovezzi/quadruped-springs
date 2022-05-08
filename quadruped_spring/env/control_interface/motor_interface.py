import numpy as np

from quadruped_spring.env.control_interface.interface_base import MotorInterfaceBase


class MotorInterfacePD(MotorInterfaceBase):
    """Command Action interface for PD motor control mode."""

    def __init__(self, robot_config):
        super().__init__(robot_config)
        self._motor_control_mode = "PD"
        self._motor_control_mode_ROB = "PD"
        self._lower_lim = self._robot_config.RL_LOWER_ANGLE_JOINT
        self._upper_lim = self._robot_config.RL_UPPER_ANGLE_JOINT
        self._init_pose = np.copy(self._robot_config.INIT_MOTOR_ANGLES)
        self._symm_idx = 0

    def _reset(self, robot):
        super()._reset(robot)
        self._init_pose = np.copy(self._robot_config.INIT_MOTOR_ANGLES)

    def _transform_action_to_motor_command(self, action):
        command = self._scale_helper_action_to_motor_command(action)
        return command

    def _transform_motor_command_to_action(self, command):
        action = self._scale_helper_motor_command_to_action(command)
        return action

    def get_landing_pose(self):
        return self._robot_config.ANGLE_LANDING_POSE


class MotorInterfaceCARTESIAN_PD(MotorInterfaceBase):
    """Command Action interface for CARTESIAN_PD motor control mode."""

    def __init__(self, robot_config):
        super().__init__(robot_config)
        self._motor_control_mode = "CARTESIAN_PD"
        self._motor_control_mode_ROB = "PD"
        self._lower_lim = self._robot_config.RL_LOWER_CARTESIAN_POS
        self._upper_lim = self._robot_config.RL_UPPER_CARTESIAN_POS
        self._init_pose = np.copy(self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME)
        self._symm_idx = 1

    def _reset(self, robot):
        super()._reset(robot)
        self._init_pose = np.copy(self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME)

    def _transform_action_to_motor_command(self, action):
        des_foot_pos = self._scale_helper_action_to_motor_command(action)
        q_des = np.array(
            list(
                map(
                    lambda i: self._robot.ComputeInverseKinematics(i, des_foot_pos[3 * i : 3 * (i + 1)]),
                    range(self._robot_config.NUM_LEGS),
                )
            )
        )
        return q_des.flatten()

    def _transform_motor_command_to_action(self, command):
        action = self._scale_helper_motor_command_to_action(command)
        return action

    def get_landing_pose(self):
        return self._robot_config.CARTESIAN_LANDING_POSE


class MotorInterfaceTORQUE(MotorInterfaceBase):
    """In order to supply pure torque to motors"""

    def __init__(self, robot_config):
        super().__init__(robot_config)
        self._motor_control_mode = "TORQUE"
        self._motor_control_mode_ROB = "TORQUE"
        self._lower_lim = -self._robot_config.TORQUE_LIMITS
        self._upper_lim = self._robot_config.TORQUE_LIMITS
        self._init_pose = np.zeros(self._robot_config.NUM_MOTORS)

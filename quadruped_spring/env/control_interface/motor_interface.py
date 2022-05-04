import numpy as np
from quadruped_spring.env.control_interface.interface_base import MotorInterfaceBase


class MotorInterfacePD(MotorInterfaceBase):
    """Command Action interface for PD motor control mode."""
    def __init__(self, robot_config):
        super().__init__(robot_config)
        self._motor_control_mode = "PD"
        self._lower_lim = self._robot_config.RL_LOWER_ANGLE_JOINT
        self._upper_lim = self._robot_config.RL_UPPER_ANGLE_JOINT
        self._symm_idx = 0
    
    def get_init_pose(self):
        return self._robot_config.INIT_MOTOR_ANGLES

    
class MotorInterfaceCARTESIAN_PD(MotorInterfaceBase):
    """Command Action interface for CARTESIAN_PD motor control mode."""
    def __init__(self, robot_config):
        super().__init__(robot_config)
        self._motor_control_mode = "CARTESIAN_PD"
        self._lower_lim = self._robot_config.RL_LOWER_CARTESIAN_POS
        self._upper_lim = self._robot_config.RL_UPPER_CARTESIAN_POS
        self._symm_idx = 1
    
    def get_init_pose(self):
        return self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME
    
class MotorInterfaceTORQUE(MotorInterfaceBase):
    """In order to supply pure torque to motors"""
    def __init__(self, robot_config):
        super().__init__(robot_config)
        self._motor_control_mode = "TORQUE"
        self._lower_lim = -self._robot_config.TORQUE_LIMITS
        self._upper_lim = self._robot_config.TORQUE_LIMITS

from mimetypes import init

from quadruped_spring.env.control_interface.action_interface import DefaultActionWrapper
from quadruped_spring.env.control_interface.motor_interface import MotorInterfacePD


def temporary_switch_motor_control_mode(env, mode="PD"):
    def aux_wrapper(foo):
        def wrapper(*args, **kwargs):
            """Temporary switch motor control mode"""
            tmp_save_motor_mode = env.robot._motor_model._motor_control_mode
            env.robot._motor_model._motor_control_mode = mode
            ret = foo(*args, **kwargs)
            env.robot._motor_model._motor_control_mode = tmp_save_motor_mode
            return ret

        return wrapper

    return aux_wrapper


def settle_robot_by_PD(env):
    """Settle robot by PD and add noise to init configuration."""
    motorPD = MotorInterfacePD(env)
    aci = DefaultActionWrapper(motorPD)
    aci._reset(env.robot)
    init_angles = aci._robot_config.INIT_MOTOR_ANGLES + aci._robot_config.JOINT_OFFSETS
    settle = temporary_switch_motor_control_mode(env, "PD")
    settle = settle(aci._settle_robot_by_reference)
    settle(init_angles, 1500)

from mimetypes import init

from quadruped_spring.env.control_interface.action_interface import DefaultActionWrapper
from quadruped_spring.env.control_interface.motor_interface import MotorInterfacePD


def temporary_switch_motor_control_mode(env, mode="PD"):
    def aux_wrapper(method):
        def wrapper(*args, **kwargs):
            """Temporary switch motor control mode"""
            tmp_save_motor_mode = env.robot._motor_model._motor_control_mode
            env.robot._motor_model._motor_control_mode = mode
            ret = method(*args, **kwargs)
            env.robot._motor_model._motor_control_mode = tmp_save_motor_mode
            return ret

        return wrapper

    return aux_wrapper


def settle_robot_by_pd(env):
    """Settle robot by PD and add noise to init configuration."""
    motorPD = MotorInterfacePD(env)
    aci = DefaultActionWrapper(motorPD)  # aci -> action control interface
    aci._reset(env.robot)
    init_angles = aci.get_init_pose()
    settle = temporary_switch_motor_control_mode(env, "PD")
    settle = settle(aci._settle_robot_by_reference)
    settle(init_angles, 1500)

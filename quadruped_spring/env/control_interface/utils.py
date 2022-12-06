import numpy as np

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


def find_config_from_height(des_height, robot):
    """Return the config such that robot mass center height == des_height"""
    link_length = robot._robot_config._THIGH_LINK_LENGTH
    q_hip = 0
    q_thigh = np.arccos(des_height / (2 * link_length))
    q_calf = -2 * q_thigh
    config = [q_hip, q_thigh, q_calf] * 4
    return config


# def config_des_phi(phi_des, robot):
#     """Return the config such that the robot has pitch orientation == phi_des"""
#     x_offset = robot._robot_config.X_OFFSET
#     q = robot.GetMotorAngles()
#     leg_front_ids = [0, 1]
#     leg_back_ids = [2, 3]
#     z_avg = 0
#     for i in leg_front_ids:
#         _, pos = robot.ComputeJacobianAndPosition(i)
#         z = -pos[2]
#         z_avg += z
#     height_front = z_avg / 2
#     z_avg = 0
#     for i in leg_back_ids:
#         q_leg = q[3 * i, 3 * (i + 1)]
#         _, pos = robot.ComputeJacobianAndPosition(i)
#         z = -pos[2]
#         z_avg += z
#     height_rear = z_avg / 2
#     actual_phi = np.arcsin((height_rear - height_front) / (2 * x_offset))
#     print(f'actual phi -> {actual_phi}')
#     delta_z = np.sin(phi_des) * 2 * x_offset
#     height_rear_des = height_rear + delta_z / 2
#     height_front_des = height_front + delta_z / 2


def get_pose_from_phi_des(phi_des, robot):
    """
    Get the pose such that the robot pitch angle is equal to phi_des.
    Please note that this method works only if the robot starts with the
    nominal position.
    Params:
    - phi_des:  the desired pitch angle.
    - robot:    the robot instance. Usefull to get robot geometric quantities
                and solve the inverse kinematic problem.
    """
    cartesian_pos_des = compute_des_feet_cartesian_pos(phi_des, robot)
    q_des = inverse_kinematics(cartesian_pos_des, robot)
    return q_des


def compute_des_feet_cartesian_pos(phi_des, robot):
    radius = robot._robot_config.X_OFFSET
    init_feet_pos, _ = robot.ComputeFeetPosAndVel()
    hip_front_des_pos = radius * np.asarray([np.cos(phi_des), -np.sin(phi_des)])
    hip_rear_des_pos = radius * np.asarray([-np.cos(phi_des), np.sin(phi_des)])
    feet_front_des_pos = [radius - hip_front_des_pos[0], 0, -hip_front_des_pos[1]]
    feet_rear_des_pos = [-radius - hip_rear_des_pos[0], 0, -hip_rear_des_pos[1]]
    return (feet_front_des_pos * 2 + feet_rear_des_pos * 2) + init_feet_pos


def inverse_kinematics(cartesian_pos_des, robot):
    q_des = np.zeros(robot._robot_config.NUM_MOTORS)
    for i in range(robot._robot_config.NUM_LEGS):
        xyz_leg = cartesian_pos_des[3 * i : 3 * (i + 1)]
        q_des[3 * i : 3 * (i + 1)] = robot.ComputeInverseKinematics(i, xyz_leg)
    return q_des

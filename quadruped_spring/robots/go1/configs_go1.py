"""Defines the GO1 robot related constants and URDF specs."""
import inspect
import os
import re

import numpy as np
import pybullet as pyb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

URDF_ROOT = parentdir
URDF_FILENAME = "go1/go1_description/urdf/go1.urdf"

##################################################################################
# Default robot configuration (i.e. base and joint positions, etc.)
##################################################################################
NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1]  # when hung up in air (for debugging)
INIT_POSITION = [0, 0, 0.33]  # normal initial height
IS_FALLEN_HEIGHT = 0.18  # height at which robot is considered fallen

INIT_ORIENTATION = (0, 0, 0, 1)
_, INIT_ORIENTATION_INV = pyb.invertTransform(position=[0, 0, 0], orientation=INIT_ORIENTATION)

# default angles (for init)
DEFAULT_HIP_ANGLE = 0  # anca
DEFAULT_THIGH_ANGLE = np.pi / 4  # coscia
DEFAULT_CALF_ANGLE = -np.pi / 2  # ginocchio

INIT_JOINT_ANGLES = np.array([DEFAULT_HIP_ANGLE, DEFAULT_THIGH_ANGLE, DEFAULT_CALF_ANGLE] * NUM_LEGS)
INIT_MOTOR_ANGLES = INIT_JOINT_ANGLES
# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# joint offsets
HIP_JOINT_OFFSET = 0.0
THIGH_JOINT_OFFSET = 0.0
CALF_JOINT_OFFSET = 0.0

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_OFFSETS = np.array([HIP_JOINT_OFFSET, THIGH_JOINT_OFFSET, CALF_JOINT_OFFSET] * NUM_LEGS)

# Kinematics
HIP_LINK_LENGTH = 0.0847
THIGH_LINK_LENGTH = 0.213
CALF_LINK_LENGTH = 0.213

NOMINAL_FOOT_POS_LEG_FRAME = np.array(
    [0, -HIP_LINK_LENGTH, -0.32, 0, HIP_LINK_LENGTH, -0.32, 0, -HIP_LINK_LENGTH, -0.32, 0, HIP_LINK_LENGTH, -0.32]
)

##################################################################################
# Actuation limits/gains, position, and velocity limits
##################################################################################
# joint limits on real system
REAL_UPPER_ANGLE_JOINT = np.array([1.0471975512, 2.96705972839, -0.837758040957] * NUM_LEGS)
REAL_LOWER_ANGLE_JOINT = np.array([-1.0471975512, -0.663225115758, -2.72271363311] * NUM_LEGS)

# modified range in simulation (min observation space for RL)
RL_UPPER_ANGLE_JOINT = np.array([0.2, DEFAULT_THIGH_ANGLE + 0.4, DEFAULT_CALF_ANGLE + 0.4] * NUM_LEGS)
RL_LOWER_ANGLE_JOINT = np.array([-0.2, DEFAULT_THIGH_ANGLE - 0.4, DEFAULT_CALF_ANGLE - 0.4] * NUM_LEGS)

# torque and velocity limits
# Set to 0.4 * ... to limit max torque
TORQUE_LIMITS = 1.0 * np.asarray([23.7, 23.7, 1.0 * 33.55] * NUM_LEGS)
VELOCITY_LIMITS = 1.0 * np.asarray([30.1, 30.1, 30.1] * NUM_LEGS)

# Sample Joint Gains
MOTOR_KP = [100.0, 100.0, 100.0] * NUM_LEGS
MOTOR_KD = [1.0, 2.0, 2.0] * NUM_LEGS

MOTOR_KP = [55, 55, 55] * NUM_LEGS
MOTOR_KD = [0.8, 0.8, 0.8] * NUM_LEGS

# Sample Cartesian Gains
kpCartesian = np.diag([500, 500, 500])
kdCartesian = np.diag([10, 10, 10])

# kpCartesian = np.diag([700, 700, 700])
# kdCartesian = np.diag([12, 12, 12])

##################################################################################
# Hip, thigh, calf strings, naming conventions from URDF (don't modify)
##################################################################################
JOINT_NAMES = (
    # front right leg
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    # front left leg
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    # rear right leg
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    # rear left leg
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
MOTOR_NAMES = JOINT_NAMES

# standard across all robots
_CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_j\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_j\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")

############################################
# Spring level joint Variables
############################################
# stiffness
K_HIP = 25
K_THIGH = 25
K_CALF = 30
# damping
D_HIP = 0.4
D_TIHGH = 0.4
D_CALF = 0.4

SPRINGS_STIFFNESS = [K_HIP, K_CALF, K_THIGH]
SPRINGS_DAMPING = [D_HIP, D_TIHGH, D_CALF]
SPRINGS_REST_ANGLE = [DEFAULT_HIP_ANGLE, DEFAULT_THIGH_ANGLE, DEFAULT_CALF_ANGLE + 0.3]

################################################
# Parameters for actions clipping
################################################
# step size is equal to 0.001 * action_repeat(default_value = 10)
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP = np.array([0.1, 0.02, 0.08])

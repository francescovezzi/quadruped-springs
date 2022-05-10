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
INIT_POSITION = [0, 0, 0.36]  # normal initial height
IS_FALLEN_HEIGHT = 0.10  # height at which robot is considered fallen

INIT_ORIENTATION = (0, 0, 0, 1)
_, INIT_ORIENTATION_INV = pyb.invertTransform(position=[0, 0, 0], orientation=INIT_ORIENTATION)

# default angles (for init)
DEFAULT_HIP_ANGLE = 0
DEFAULT_THIGH_ANGLE = np.pi / 4
DEFAULT_CALF_ANGLE = -np.pi / 2

ANGLE_LANDING_POSE = np.array([DEFAULT_HIP_ANGLE, DEFAULT_THIGH_ANGLE, DEFAULT_CALF_ANGLE] * NUM_LEGS)

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

# default foot pos in leg frame
DEFAULT_X = 0
DEFAULT_Y = HIP_LINK_LENGTH
DEFAULT_Z = -0.32
LANDING_Z = -0.29

NOMINAL_FOOT_POS_LEG_FRAME = np.array(
    list(map(lambda sign: [DEFAULT_X, sign * DEFAULT_Y, DEFAULT_Z], [-1, 1, -1, 1]))
).flatten()
CARTESIAN_LANDING_POSE = np.array(list(map(lambda sign: [DEFAULT_X, sign * DEFAULT_Y, LANDING_Z], [-1, 1, -1, 1]))).flatten()

##################################################################################
# Actuation limits/gains, position, and velocity limits
##################################################################################
# joint limits on real system
REAL_UPPER_ANGLE_JOINT = np.array([1.0471975512, 2.96705972839, -0.837758040957] * NUM_LEGS)
REAL_LOWER_ANGLE_JOINT = np.array([-1.0471975512, -0.663225115758, -2.72271363311] * NUM_LEGS)

# modified range in simulation (min observation space for RL)
RL_UPPER_ANGLE_JOINT = np.array([0.2, DEFAULT_THIGH_ANGLE + 0.5, -0.95] * NUM_LEGS)
RL_LOWER_ANGLE_JOINT = np.array(
    [-0.2, DEFAULT_THIGH_ANGLE - 0.5, -2.42] * NUM_LEGS
)  # if calf angle=-2.42 the robot height is 0.15
# RL_LOWER_ANGLE_JOINT = np.array([-0.2, DEFAULT_THIGH_ANGLE - 0.4, DEFAULT_CALF_ANGLE - 0.85] * NUM_LEGS)

RL_UPPER_CARTESIAN_POS = (
    NOMINAL_FOOT_POS_LEG_FRAME + np.array(list(map(lambda sign: [0.2, sign * 0.05, 0.18], [1, 1, 1, 1]))).flatten()
)

RL_LOWER_CARTESIAN_POS = (
    NOMINAL_FOOT_POS_LEG_FRAME - np.array(list(map(lambda sign: [0.2, sign * 0.05, 0.07], [1, 1, 1, 1]))).flatten()
)

# torque and velocity limits
# Set to 0.4 * ... to limit max torque
TORQUE_LIMITS = 1.0 * np.asarray([23.7, 23.7, 1.0 * 33.55] * NUM_LEGS)
VELOCITY_LIMITS = 1.0 * np.asarray([30.1, 30.1, 30.1] * NUM_LEGS)
RL_VELOCITY_LIMITS = 1.0 * np.asarray([10, 10, 10] * NUM_LEGS)  #  Used for noise observation calculation

# Sample Joint Gains
MOTOR_KP = [100.0, 100.0, 100.0] * NUM_LEGS
MOTOR_KD = [1.0, 2.0, 2.0] * NUM_LEGS

# Sample Cartesian Gains
kpCartesian = np.diag([1200, 2000, 2000])
kdCartesian = np.diag([13, 15, 15])

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

SPRINGS_STIFFNESS = [K_HIP, K_THIGH, K_CALF]
SPRINGS_DAMPING = [D_HIP, D_TIHGH, D_CALF]
SPRINGS_REST_ANGLE = [DEFAULT_HIP_ANGLE, DEFAULT_THIGH_ANGLE, DEFAULT_CALF_ANGLE + 0.3]

INIT_JOINT_ANGLES = np.array(SPRINGS_REST_ANGLE * NUM_LEGS)
INIT_MOTOR_ANGLES = INIT_JOINT_ANGLES

################################################
# Parameters for actions clipping
################################################
# step size is equal to 0.001 * action_repeat(default_value = 10)
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP = np.array([0.1, 0.02, 0.08])

################################################
# Sensor High Limits
################################################

VEL_LIN_HIGH = np.array([5.0] * 3)
ORIENT_RPY_HIGH = np.array([np.pi] * 3)
ORIENT_RATE_HIGH = np.array([5.0] * 3)
IMU_HIGH = np.concatenate((VEL_LIN_HIGH, ORIENT_RPY_HIGH, ORIENT_RATE_HIGH))
JOINT_ANGLES_HIGH = RL_UPPER_ANGLE_JOINT
JOINT_VELOCITIES_HIGH = RL_VELOCITY_LIMITS
CONTACT_FORCE_HIGH = np.array([5.0] * NUM_LEGS)
CONTACT_BOOL_HIGH = np.array([1.0] * NUM_LEGS)
FEET_POS_HIGH = RL_UPPER_CARTESIAN_POS
FEET_VEL_HIGH = np.array([10.0] * NUM_MOTORS)

################################################
# Sensor Low Limits
################################################

IMU_LOW = -IMU_HIGH
JOINT_ANGLES_LOW = RL_LOWER_ANGLE_JOINT
JOINT_VELOCITIES_LOW = -JOINT_VELOCITIES_HIGH
CONTACT_FORCE_LOW = -CONTACT_FORCE_HIGH
CONTACT_BOOL_LOW = np.array([0.0] * NUM_LEGS)
FEET_POS_LOW = RL_LOWER_CARTESIAN_POS
FEET_VEL_LOW = -FEET_POS_HIGH

################################################
# Sensor Noise std
################################################

STD_COEFF = 0.01

VEL_LIN_NOISE = VEL_LIN_HIGH * STD_COEFF
ORIENT_RPY_NOISE = ORIENT_RPY_HIGH * STD_COEFF
ORIENT_RATE_NOISE = ORIENT_RATE_HIGH * STD_COEFF
IMU_NOISE = np.concatenate((VEL_LIN_NOISE, ORIENT_RPY_NOISE, ORIENT_RATE_NOISE))
JOINT_ANGLES_NOISE = np.maximum(abs(JOINT_ANGLES_HIGH), abs(JOINT_ANGLES_LOW)) * STD_COEFF
JOINT_VELOCITIES_NOISE = JOINT_VELOCITIES_HIGH * STD_COEFF
CONTACT_FORCE_NOISE = CONTACT_FORCE_HIGH * STD_COEFF
CONTACT_BOOL_NOISE = np.array([0] * NUM_LEGS)
FEET_POS_NOISE = np.array([0.1, 0.05, 0.1] * NUM_LEGS) * STD_COEFF
FEET_VEL_NOISE = FEET_VEL_HIGH * STD_COEFF

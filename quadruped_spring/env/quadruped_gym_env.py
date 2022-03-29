"""This file implements the gym environment for a quadruped. """
import inspect
import os

# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import datetime
import time

# gym
import gym
import numpy as np

# pybullet
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import quadruped
from gym import spaces
from gym.utils import seeding
from scipy.spatial.transform import Rotation as R

import quadruped_spring.robots.a1.configs_a1 as a1_config
import quadruped_spring.robots.go1.configs_go1 as go1_config
from quadruped_spring.utils import action_filter

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
VIDEO_LOG_DIRECTORY = "videos/" + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")

# Implemented observation spaces for deep reinforcement learning:
#   "DEFAULT":    motor angles and velocities, body orientation
#   "LR_COURSE_OBS":  [TODO: what should you include? what is reasonable to measure on the real system?]

# Tasks to be learned with reinforcement learning
#     - "FWD_LOCOMOTION"
#         reward forward progress only
#     - "LR_COURSE_TASK"
#         [TODO: what should you train for?]
#         Ideally we want to command GO1 to run in any direction while expending minimal energy
#         It is suggested to first train to run at 3 sample velocities (0.5 m/s, 1 m/s, 1.5 m/s)
#         How will you construct your reward function?
#     - "JUMPING_TASK"
#         Sparse reward, maximizing flight time + bonus forward_distance + malus on crashing
#     - "JUMPING_ON_PLACE_TASK"
#         Sparse reward, maximizing flight time + bonus maintin base position +
#         malus on crashing + malus on not allowed contacts

# Motor control modes:
#   - "TORQUE":
#         supply raw torques to each motor (12)
#   - "PD":
#         supply desired joint positions to each motor (12)
#         torques are computed based on the joint position/velocity error
#   - "CARTESIAN_PD":
#         supply desired foot positions for each leg (12)
#         torques are computed based on the foot position/velocity error


EPISODE_LENGTH = 10  # how long before we reset the environment (max episode length for RL)
MAX_FWD_VELOCITY = 5  # to avoid exploiting simulator dynamics, cap max reward for body velocity

ROBOT_CLASS_MAP = {"A1": a1_config, "GO1": go1_config}


class QuadrupedGymEnv(gym.Env):
    """The gym environment for a quadruped {Unitree GO1}.

    It simulates the locomotion of a quadrupedal robot.
    The state space, action space, and reward functions can be chosen with:
    observation_space_mode, motor_control_mode, task_env.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        robot_model="GO1",
        isRLGymInterface=True,
        time_step=0.001,
        action_repeat=10,
        distance_weight=1e3,
        energy_weight=1e-4,  # 0.008,
        motor_control_mode="PD",
        task_env="FWD_LOCOMOTION",
        observation_space_mode="DEFAULT",
        on_rack=False,
        render=False,
        record_video=False,
        add_noise=True,
        enable_springs=False,
        enable_action_interpolation=False,
        enable_action_filter=False,
        enable_action_clipping=False,
        test_env=False,  # NOT ALLOWED FOR TRAINING!
    ):
        """Initialize the quadruped gym environment.

        Args:
          robot_model: String representing the robot model. Select between "A1" or "GO1".
          isRLGymInterface: If the gym environment is being run as RL or not. Affects
            if the actions should be scaled.
          time_step: Simulation time step.
          action_repeat: The number of simulation steps where the same actions are applied.
          distance_weight: The weight of the distance term in the reward.
          energy_weight: The weight of the energy term in the reward.
          motor_control_mode: Whether to use torque control, PD, control, etc.
          task_env: Task trying to learn (fwd locomotion, standup, etc.)
          observation_space_mode: what should be in here? Check available functions in quadruped.py
          on_rack: Whether to place the quadruped on rack. This is only used to debug
            the walking gait. In this mode, the quadruped's base is hanged midair so
            that its walking gait is clearer to visualize.
          render: Whether to render the simulation.
          record_video: Whether to record a video of each trial.
          add_noise: vary coefficient of friction
          test_env: add random terrain
          enable_springs: Whether to enable springs or not
          enable_action_interpolation: Whether to interpolate the current action
            with the previous action in order to produce smoother motions
          enable_action_filter: Boolean specifying if a lowpass filter should be
            used to smooth actions.
          enable_action_clipping: Boolean specifying if motor commands should be
            clipped or not. It's not implemented for pure torque control.
        """
        self.seed()
        try:
            robot_config = ROBOT_CLASS_MAP[robot_model]
        except KeyError:
            raise KeyError('Robot model should be "A1 or "GO1"')
        self._robot_config = robot_config
        self._isRLGymInterface = isRLGymInterface
        self._time_step = time_step
        self._action_repeat = action_repeat
        self._distance_weight = distance_weight
        self._energy_weight = energy_weight
        self._motor_control_mode = motor_control_mode
        self._TASK_ENV = task_env
        self._observation_space_mode = observation_space_mode
        self._hard_reset = True  # must fully reset simulation at init
        self._on_rack = on_rack
        self._is_render = render
        self._is_record_video = record_video
        self._add_noise = add_noise
        self._enable_springs = enable_springs
        self._enable_action_interpolation = enable_action_interpolation
        self._enable_action_filter = enable_action_filter
        self._enable_action_clipping = enable_action_clipping
        self._using_test_env = test_env
        if test_env:
            self._add_noise = True
            self._observation_noise_stdev = 0.01  # TODO: check if increasing makes sense
        else:
            self._observation_noise_stdev = 0.01

        # other bookkeeping
        self._num_bullet_solver_iterations = int(300 / action_repeat)
        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = np.array([0, 0, 0])
        self._last_frame_time = 0.0  # for rendering
        self._MAX_EP_LEN = EPISODE_LENGTH  # max sim time in seconds, arbitrary
        self._action_bound = 1.0

        self.setupActionSpace()
        self.setupObservationSpace()
        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()

        if self._enable_action_filter:
            self._action_filter = self._build_action_filter()

        self.videoLogID = None
        self.reset()

    ######################################################################################
    # RL Observation and Action spaces
    ######################################################################################
    def setupObservationSpace(self):
        # NOTE:
        # The observation space is actually used to calulate the observation noise.
        # It is proportional to obs_high
        if self._observation_space_mode == "DEFAULT":
            obs_high, obs_low = self._set_obs_space_default()
        elif self._observation_space_mode == "LR_COURSE_OBS":
            obs_high, obs_low = self._set_obs_space_lr_course()
        elif self._observation_space_mode == "JUMPING_ON_PLACE_OBS":
            obs_high, obs_low = self._set_obs_space_jump_on_place()
        else:
            raise ValueError(f"observation space {self._observation_space_mode} not defined or not intended")

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def _set_obs_space_jump_on_place(self):
        q_high = self._robot_config.RL_UPPER_ANGLE_JOINT
        q_low = self._robot_config.RL_LOWER_ANGLE_JOINT
        dq_high = self._robot_config.VELOCITY_LIMITS
        dq_low = -self._robot_config.VELOCITY_LIMITS
        vel_high = np.array([MAX_FWD_VELOCITY] * 3)
        vel_low = np.array([-MAX_FWD_VELOCITY] * 3)
        rpy_high = np.array([np.pi] * 3)
        rpy_low = np.array([-np.pi] * 3)
        drpy_high = np.array([5.0] * 3)
        drpy_low = np.array([-5.0] * 3)
        foot_pos_high = np.array([0.1, 0.05, 0.1])
        foot_pos_low = -foot_pos_high
        foot_vel_high = np.array([10.0] * 12)
        foot_vel_low = np.array([-10.0] * 12)
        contact_high = np.array([1.0] * 4)
        contact_low = np.array([0.0] * 4)

        observation_high = (
            np.concatenate(
                (
                    vel_high,
                    rpy_high,
                    drpy_high,
                    foot_pos_high,
                    foot_pos_high,
                    foot_pos_high,
                    foot_pos_high,
                    foot_vel_high,
                    contact_high,
                )
            )
            + OBSERVATION_EPS
        )
        observation_low = (
            np.concatenate(
                (
                    vel_low,
                    rpy_low,
                    drpy_low,
                    foot_pos_low,
                    foot_pos_low,
                    foot_pos_low,
                    foot_pos_low,
                    foot_vel_low,
                    contact_low,
                )
            )
            - OBSERVATION_EPS
        )

        return observation_high, observation_low

    def _set_obs_space_default(self):
        observation_high = (
            np.concatenate((self._robot_config.RL_UPPER_ANGLE_JOINT, self._robot_config.VELOCITY_LIMITS, np.array([1.0] * 4)))
            + OBSERVATION_EPS
        )
        observation_low = (
            np.concatenate(
                (self._robot_config.RL_LOWER_ANGLE_JOINT, -self._robot_config.VELOCITY_LIMITS, np.array([-1.0] * 4))
            )
            - OBSERVATION_EPS
        )
        return observation_high, observation_low

    def _set_obs_space_lr_course(self):
        q_high = self._robot_config.RL_UPPER_ANGLE_JOINT
        q_low = self._robot_config.RL_LOWER_ANGLE_JOINT
        dq_high = self._robot_config.VELOCITY_LIMITS
        dq_low = -self._robot_config.VELOCITY_LIMITS
        vel_high = np.array([MAX_FWD_VELOCITY] * 3)
        vel_low = np.array([-MAX_FWD_VELOCITY] * 3)
        rpy_high = np.array([np.pi] * 3)
        rpy_low = np.array([-np.pi] * 3)
        drpy_high = np.array([5.0] * 3)
        drpy_low = np.array([-5.0] * 3)
        foot_pos_high = np.array([0.1, 0.05, 0.1])
        foot_pos_low = -foot_pos_high
        foot_vel_high = np.array([10.0] * 12)
        foot_vel_low = np.array([-10.0] * 12)
        contact_high = np.array([1.0] * 4)
        contact_low = np.array([0.0] * 4)

        observation_high = (
            np.concatenate(
                (
                    vel_high,
                    vel_high[:2],
                    rpy_high,
                    drpy_high,
                    foot_pos_high,
                    foot_pos_high,
                    foot_pos_high,
                    foot_pos_high,
                    foot_vel_high,
                    contact_high,
                )
            )
            + OBSERVATION_EPS
        )
        observation_low = (
            np.concatenate(
                (
                    vel_low,
                    vel_low[:2],
                    rpy_low,
                    drpy_low,
                    foot_pos_low,
                    foot_pos_low,
                    foot_pos_low,
                    foot_pos_low,
                    foot_vel_low,
                    contact_low,
                )
            )
            - OBSERVATION_EPS
        )

        return observation_high, observation_low

    def setupActionSpace(self):
        """Set up action space for RL."""
        if self._motor_control_mode in ["PD", "TORQUE", "CARTESIAN_PD"]:
            action_dim = 12
        else:
            raise ValueError("motor control mode " + self._motor_control_mode + " not implemented yet.")
        action_high = np.array([1] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self._action_dim = action_dim

    def _get_observation(self):
        """Get observation, depending on obs space selected."""
        if self._observation_space_mode == "DEFAULT":
            self._get_obs_default()
        elif self._observation_space_mode == "LR_COURSE_OBS":
            self._get_obs_lr_course()
        elif self._observation_space_mode == "JUMPING_ON_PLACE_OBS":
            self._get_obs_jump_on_place()
        else:
            raise ValueError("observation space not defined or not intended")

        self._add_obs_noise = (
            np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) * self.observation_space.high
        )
        return self._observation

    def _get_obs_jump_on_place(self):
        q = self.robot.GetMotorAngles()
        vel = self.robot.GetBaseLinearVelocity()
        rpy = self.robot.GetBaseOrientationRollPitchYaw()
        drpy = self.robot.GetTrueBaseRollPitchYawRate()
        foot_pos, foot_vel = self.robot.ComputeFeetPosAndVel()

        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = self.robot.GetContactInfo()

        self._observation = np.concatenate((vel, rpy, drpy, foot_pos, foot_vel, feetInContactBool))

    def _get_obs_default(self):
        self._observation = np.concatenate(
            (self.robot.GetMotorAngles(), self.robot.GetMotorVelocities(), self.robot.GetBaseOrientation())
        )

    def _get_obs_lr_course(self):
        # q = self.robot.GetMotorAngles()
        dq = self.robot.GetMotorVelocities()
        vel = self.robot.GetBaseLinearVelocity()
        rpy = self.robot.GetBaseOrientationRollPitchYaw()
        drpy = self.robot.GetTrueBaseRollPitchYawRate()
        foot_pos = np.zeros(12)
        foot_vel = np.zeros(12)
        for i in range(4):
            dq_i = dq[3 * i : 3 * (i + 1)]
            J, xyz = self.robot.ComputeJacobianAndPosition(i)
            foot_pos[3 * i : 3 * (i + 1)] = xyz
            foot_vel[3 * i : 3 * (i + 1)] = J @ dq_i

        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = self.robot.GetContactInfo()

        self._observation = np.concatenate((vel, self._v_des, rpy, drpy, foot_pos, foot_vel, feetInContactBool))

    def _noisy_observation(self):
        self._get_observation()
        observation = np.array(self._observation)
        if self._observation_noise_stdev > 0:
            observation += self._add_obs_noise
        return observation

    ######################################################################################
    # Termination and reward
    ######################################################################################
    def is_fallen(self, dot_prod_min=0.85):
        """Decide whether the quadruped has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the quadruped is considered fallen.

        Returns:
          Boolean value that indicates whether the quadruped has fallen.
        """
        base_rpy = self.robot.GetBaseOrientationRollPitchYaw()
        orientation = self.robot.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.robot.GetBasePosition()
        return (
            np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < dot_prod_min or pos[2] < self._robot_config.IS_FALLEN_HEIGHT
        )

    def _not_allowed_contact(self):
        """
        Return True if the robot is performing some not allowed contact
        as touching the ground with knees
        """
        _, num_invalid_contacts, _, _ = self.robot.GetContactInfo()

        return num_invalid_contacts

    def _termination(self):
        """Decide whether we should stop the episode and reset the environment."""
        if self._TASK_ENV in ["JUMPING_TASK", "LR_COURSE_TASK", "FWD_LOCOMOTION"]:
            return self.is_fallen()
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            return self.is_fallen() or self._not_allowed_contact()
        else:
            raise ValueError("This task mode is not implemented yet.")

    def _reward_jumping(self):
        # Change is fallen height
        self._robot_config.IS_FALLEN_HEIGHT = 0.01

        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        flight_time_reward = 0.0
        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
            else:
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        return flight_time_reward
        # max_height_reward = self.robot.GetBasePosition()[2] / self._init_height
        # only_positive_height = max(self.robot.GetBasePosition()[2] - self._init_height, 0.0)
        # max_height_reward = only_positive_height ** 2 / self._init_height ** 2
        # return max_height_reward + flight_time_reward
        # return max_height_reward + no_feet_in_contact_reward

    def _reward_jumping_on_place(self):
        """Reward maximum flight time"""
        # Change is fallen height
        self._robot_config.IS_FALLEN_HEIGHT = 0.01

        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        flight_time_reward = 0.0
        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
                self._robot_orientation_take_off_quat = np.array(self.robot.GetBaseOrientation())
            else:
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        _, _, yaw = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_yaw = max(np.abs(yaw), self._max_yaw)

        return flight_time_reward

    def _reward_fwd_locomotion(self):
        """Reward progress in the positive world x direction."""
        current_base_position = self.robot.GetBasePosition()
        forward_reward = current_base_position[0] - self._last_base_position[0]
        self._last_base_position = current_base_position
        # clip reward to MAX_FWD_VELOCITY (avoid exploiting simulator dynamics)
        if MAX_FWD_VELOCITY < np.inf:
            # calculate what max distance can be over last time interval based on max allowed fwd velocity
            max_dist = MAX_FWD_VELOCITY * (self._time_step * self._action_repeat)
            forward_reward = min(forward_reward, max_dist)

        v_x = max(min(self.robot.GetBaseLinearVelocity()[0], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
        v_y = max(min(self.robot.GetBaseLinearVelocity()[1], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
        v = np.array([v_x, v_y])

        return 1e-2 * np.linalg.norm(v)

    def _reward_lr_course(self):
        """Implement your reward function here. How will you improve upon the above?"""
        current_base_position = np.array(self.robot.GetBasePosition())
        v_ = (current_base_position[:2] - self._last_base_position[:2]) / (self._time_step * self._action_repeat)
        self._last_base_position = current_base_position
        # clip reward to MAX_FWD_VELOCITY (avoid exploiting simulator dynamics)
        if MAX_FWD_VELOCITY < np.inf:
            v_x = max(min(v_[0], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
            v_y = max(min(v_[1], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
            v = np.array([v_x, v_y])

        e = self._v_des - v

        v_norm = max(np.linalg.norm(v), 0.001)  # prevent CoT explosion for very small movements
        P = 0.0
        for i in range(len(self._dt_motor_torques)):
            P += np.array(self._dt_motor_torques[i]).dot(np.array(self._dt_motor_velocities[i]))
        P = P / len(self._dt_motor_torques)
        CoT = P / (5.0 * 10.0 * v_norm)

        ddq = (np.array(self._dt_motor_velocities[-1]) - np.array(self._dt_motor_velocities[0])) / (
            self._time_step * self._action_repeat
        )

        rpy = self.robot.GetBaseOrientationRollPitchYaw()

        r = (
            0.1 * v_norm
            - 0.0001 * abs(CoT)
            + 0.1 * np.exp(-1.0 * ddq.dot(ddq))
            - 0.05 * abs(rpy[2])
            + np.exp(-10.0 * e.dot(e))
        )

        return r

    def _reward(self):
        """Get reward depending on task"""
        if self._TASK_ENV == "FWD_LOCOMOTION":
            return self._reward_fwd_locomotion()
        elif self._TASK_ENV == "LR_COURSE_TASK":
            return self._reward_lr_course()
        elif self._TASK_ENV == "JUMPING_TASK":
            return self._reward_jumping()
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            return self._reward_jumping_on_place()
        else:
            raise ValueError("This task mode is not implemented yet.")

    def _reward_end_episode(self, reward):
        """add bonus and malus at the end of the episode"""
        if self._TASK_ENV == "JUMPING_TASK":
            return self._reward_end_jumping(reward)
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            return self._reward_end_jumping_on_place(reward)
        else:
            # do nothing
            return reward

    def _reward_end_jumping(self, reward):
        """Add bonus and malus at the end of the episode for jumping task"""
        if self._termination():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        reward += self._max_flight_time
        max_distance = 0.2
        # Normalize forward distance reward
        reward += 0.1 * self._max_forward_distance / max_distance
        if self._max_flight_time > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * self._max_flight_time
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward

    def _reward_end_jumping_on_place(self, reward):
        """Add bonus and malus at the end of the episode for jumping on place task"""
        if self._termination():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        reward += self._max_flight_time
        reward += 0.05 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += 0.1 * np.exp(-self._max_forward_distance**2 / 0.05)  # be on place

        if self._max_flight_time > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * self._max_flight_time
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward

    ######################################################################################
    # Step simulation, map policy network actions to joint commands, etc.
    ######################################################################################
    def _interpolate_actions(self, action, substep_count):
        """If enabled, interpolates between the current and previous actions.

        Args:
        action: current action.
        substep_count: the step count should be between [0, self.__action_repeat).

        Returns:
        If interpolation is enabled, returns interpolated action depending on
        the current action repeat substep.
        """
        if self._enable_action_interpolation and self._last_action is not None:
            interp_fraction = float(substep_count + 1) / self._action_repeat
            interpolated_action = self._last_action + interp_fraction * (action - self._last_action)
        else:
            interpolated_action = action

        return interpolated_action

    def _transform_action_to_motor_command(self, action):
        """Map actions from RL (i.e. in [-1,1]) to joint commands based on motor_control_mode."""
        # clip actions to action bounds
        action = np.clip(action, -self._action_bound - ACTION_EPS, self._action_bound + ACTION_EPS)
        if self._motor_control_mode == "PD":
            action = self._scale_helper(
                action, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT
            )
            action = np.clip(action, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT)
            if self._enable_action_clipping:
                action = self._clip_motor_commands(action)
        elif self._motor_control_mode == "CARTESIAN_PD":
            action = self.ScaleActionToCartesianPos(action)
            # Here the clipping happens inside ScaleActionToCartesianPos
        else:
            raise ValueError("RL motor control mode" + self._motor_control_mode + "not implemented yet.")
        return action

    def _scale_helper(self, action, lower_lim, upper_lim):
        """Helper to linearly scale from [-1,1] to lower/upper limits."""
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        return np.clip(new_a, lower_lim, upper_lim)

    def ScaleActionToCartesianPos(self, actions):
        """Scale RL action to Cartesian PD ranges.
        Edit ranges, limits etc., but make sure to use Cartesian PD to compute the torques.
        """
        # clip RL actions to be between -1 and 1 (standard RL technique)
        # print(actions)
        u = np.clip(actions, -1, 1)
        # scale to corresponding desired foot positions (i.e. ranges in x,y,z we allow the agent to choose foot positions)
        # [TODO: edit (do you think these should these be increased? How limiting is this?)]
        scale_array = np.array([0.2, 0.05, 0.15] * 4)  # [0.1, 0.05, 0.08]*4)
        # add to nominal foot position in leg frame (what are the final ranges?)
        des_foot_pos = self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME + scale_array * u
        # get Cartesian kp and kd gains (can be modified)
        kpCartesian = self._robot_config.kpCartesian
        kdCartesian = self._robot_config.kdCartesian
        # get current motor velocities
        dq = self.robot.GetMotorVelocities()

        action = np.zeros(12)
        for i in range(4):
            # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
            J, xyz = self.robot.ComputeJacobianAndPosition(i)
            # Get current foot velocity in leg frame (Equation 2)
            dxyz = J @ dq[3 * i : 3 * (i + 1)]
            delta_foot_pos = xyz - des_foot_pos[3 * i : 3 * (i + 1)]

            # clamp the motor command by the joint limit, in case weired things happens
            if self._enable_action_clipping:
                delta_foot_pos = np.clip(
                    delta_foot_pos,
                    -self._robot_config.MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP,
                    self._robot_config.MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP,
                )

            F_foot = -kpCartesian @ delta_foot_pos - kdCartesian @ dxyz
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau = J.T @ F_foot
            action[3 * i : 3 * (i + 1)] = tau  # TODO: add white noise
        return action

    def _map_command_to_action(self, command):
        """
        Given the desired motor commmand returns the action that produces
        that motor command
        """
        if self._motor_control_mode == "PD":
            max_angles = self._robot_config.RL_UPPER_ANGLE_JOINT
            min_angles = self._robot_config.RL_LOWER_ANGLE_JOINT
            command = np.clip(command, min_angles, max_angles)
            action = 2 * (command - min_angles) / (max_angles - min_angles) - 1
        elif self._motor_control_mode == "CARTESIAN_PD":
            raise ValueError(
                f"for the motor control mode {self._motor_control_mode} the mapping from foot cartesian position to action not implemented yet."
            )
        else:
            raise ValueError(f"The motor control mode {self._motor_control_mode} is not implemented for RL")

        action = np.clip(action, -1, 1)
        command = self._transform_action_to_motor_command(action)
        return action

    def _clip_motor_commands(self, des_angles):
        """Clips motor commands.

        Args:
        des_angles: np.array. They are the desired motor angles (for PD control only)

        Returns:
        Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        if self._motor_control_mode == "PD":
            max_angle_change = self._robot_config.MAX_MOTOR_ANGLE_CHANGE_PER_STEP
            current_motor_angles = self.robot.GetMotorAngles()
            motor_commands = np.clip(
                des_angles, current_motor_angles - max_angle_change, current_motor_angles + max_angle_change
            )
            return motor_commands
        else:
            raise ValueError(f"Clipping angles available for PD control only, not in {self._motor_control_mode}")

    def step(self, action):
        """Step forward the simulation, given the action."""
        curr_act = action.copy()
        if self._enable_action_filter:
            curr_act = self._filter_action(curr_act)
        # save motor torques and velocities to compute power in reward function
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        for sub_step in range(self._action_repeat):
            if self._isRLGymInterface:
                if self._enable_action_interpolation:
                    curr_act = self._interpolate_actions(action, sub_step)
                proc_action = self._transform_action_to_motor_command(curr_act)
            else:
                proc_action = curr_act
            self.robot.ApplyAction(proc_action)
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            self._dt_motor_torques.append(self.robot.GetMotorTorques())
            self._dt_motor_velocities.append(self.robot.GetMotorVelocities())

            if self._is_render:
                self._render_step_helper()

        self._last_action = curr_act
        self._env_step_counter += 1
        reward = self._reward()
        done = False
        infos = {"base_pos": self.robot.GetBasePosition()}

        if self._termination() or self.get_sim_time() > self._MAX_EP_LEN:
            infos["TimeLimit.truncated"] = not self._termination()
            done = True

        # Update the actual reward at the end of the episode with bonus or malus
        if done:
            reward = self._reward_end_episode(reward)

        return np.array(self._noisy_observation()), reward, done, infos

    ###################################################
    # Filtering to smooth actions
    ###################################################
    def _build_action_filter(self):
        sampling_rate = 1 / (self._time_step * self._action_repeat)
        num_joints = self._action_dim
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
        return a_filter

    def _reset_action_filter(self):
        self._action_filter.reset()

    def _filter_action(self, action):
        filtered_action = self._action_filter.filter(action)
        return filtered_action

    def _init_filter(self):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        default_action = self._compute_first_actions()
        self._action_filter.init_history(default_action)

    def _compute_first_actions(self):
        if self._motor_control_mode == "PD":
            init_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
            default_action = self._map_command_to_action(init_angles)
        elif self._motor_control_mode == "CARTESIAN_PD":
            # go toward NOMINAL_FOOT_POS_LEG_FRAME
            default_action = np.array([0, 0, 0] * self._robot_config.NUM_LEGS)
        else:
            raise ValueError(f"The motor control mode {self._motor_control_mode} is not implemented for RL")

        return np.clip(default_action, -1, 1)

    ######################################################################################
    # Reset
    ######################################################################################
    def reset(self):
        """Set up simulation environment."""
        mu_min = 0.5
        if self._hard_reset:
            # set up pybullet simulation
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self.plane = self._pybullet_client.loadURDF(
                pybullet_data.getDataPath() + "/plane.urdf", basePosition=[80, 0, 0]
            )  # to extend available running space (shift)
            self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._pybullet_client.setGravity(0, 0, -9.8)
            self.robot = quadruped.Quadruped(
                pybullet_client=self._pybullet_client,
                robot_config=self._robot_config,
                motor_control_mode=self._motor_control_mode,
                on_rack=self._on_rack,
                render=self._is_render,
                enable_springs=self._enable_springs,
            )

            if self._add_noise:
                ground_mu_k = mu_min + (1 - mu_min) * np.random.random()
                self._ground_mu_k = ground_mu_k
                self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
                if self._is_render:
                    print("ground friction coefficient is", ground_mu_k)

            if self._using_test_env:
                pass
                # self.add_random_boxes()
                # self._add_base_mass_offset()
        else:
            self.robot.Reset(reload_urdf=False)

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._init_task_variables()

        if self._is_render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])

        if self._enable_action_filter:
            self._reset_action_filter()
            self._init_filter()

        self._settle_robot()  # Settle robot after being spawned

        self._last_action = np.zeros(self._action_dim)
        if self._is_record_video:
            self.recordVideoHelper()
        return self._noisy_observation()

    def _settle_robot_by_action(self):
        """Settle robot in according to the used motor control mode in RL interface"""
        init_action = self._compute_first_actions()
        if self._is_render:
            time.sleep(0.2)
        for _ in range(1000):
            proc_action = self._transform_action_to_motor_command(init_action)
            self.robot.ApplyAction(proc_action)
            if self._is_render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()

    def _settle_robot_by_PD(self):
        """Settle robot and add noise to init configuration."""
        # change to PD control mode to set initial position, then set back..
        tmp_save_motor_control_mode_ENV = self._motor_control_mode
        tmp_save_motor_control_mode_ROB = self.robot._motor_control_mode
        self._motor_control_mode = "PD"
        self.robot._motor_control_mode = "PD"
        try:
            tmp_save_motor_control_mode_MOT = self.robot._motor_model._motor_control_mode
            self.robot._motor_model._motor_control_mode = "PD"
        except:
            pass
        init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
        if self._is_render:
            time.sleep(0.2)
        for _ in range(800):
            self.robot.ApplyAction(init_motor_angles)
            if self._is_render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()

        # set control mode back
        self._motor_control_mode = tmp_save_motor_control_mode_ENV
        self.robot._motor_control_mode = tmp_save_motor_control_mode_ROB
        try:
            self.robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
        except:
            pass

    def _settle_robot(self):
        if self._isRLGymInterface:
            self._settle_robot_by_action()
        else:
            self._settle_robot_by_PD()

    def _init_task_variables(self):
        if self._TASK_ENV == "JUMPING_TASK":
            self._init_variables_jumping()
        elif self._TASK_ENV in ["LR_COURSE_TASK", "FWD_LOCOMOTION"]:
            self._v_des = np.array([2.0, 0.0])  # (np.random.uniform(), 2.*np.random.uniform()-1.)
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            self._init_variables_jumping_on_place()
        else:
            pass

    def _init_variables_jumping(self):
        # For the jumping task
        self._v_des = np.array([2.0, 0.0])  # (np.random.uniform(), 2.*np.random.uniform()-1.)
        self._init_height = self.robot.GetBasePosition()[2]
        self._all_feet_in_the_air = False
        self._time_take_off = self.get_sim_time()
        self._robot_pose_take_off = self.robot.GetBasePosition()
        self._robot_orientation_take_off = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0

    def _init_variables_jumping_on_place(self):
        self._v_des = np.array([0.0, 0.0])
        self._init_height = self.robot.GetBasePosition()[2]
        self._all_feet_in_the_air = False
        self._time_take_off = self.get_sim_time()
        self._robot_pose_take_off = self.robot.GetBasePosition()
        self._robot_orientation_take_off = self.robot.GetBaseOrientationRollPitchYaw()
        self._robot_orientation_take_off_quat = self.robot.GetBaseOrientation()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_yaw = 0.0

    ######################################################################################
    # Render, record videos, bookkeping, and misc pybullet helpers.
    ######################################################################################
    def startRecordingVideo(self, name):
        self.videoLogID = self._pybullet_client.startStateLogging(self._pybullet_client.STATE_LOGGING_VIDEO_MP4, name)

    def stopRecordingVideo(self):
        self._pybullet_client.stopStateLogging(self.videoLogID)

    def close(self):
        if self._is_record_video:
            self.stopRecordingVideo()
        self._pybullet_client.disconnect()

    def recordVideoHelper(self, extra_filename=None):
        """Helper to record video, if not already, or end and start a new one"""
        # If no ID, this is the first video, so make a directory and start logging
        if self.videoLogID == None:
            directoryName = VIDEO_LOG_DIRECTORY
            assert isinstance(directoryName, str)
            os.makedirs(directoryName, exist_ok=True)
            self.videoDirectory = directoryName
        else:
            # stop recording and record a new one
            self.stopRecordingVideo()

        if extra_filename is not None:
            output_video_filename = (
                self.videoDirectory
                + "/"
                + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")
                + extra_filename
                + ".MP4"
            )
        else:
            output_video_filename = (
                self.videoDirectory + "/" + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"
            )
        logID = self.startRecordingVideo(output_video_filename)
        self.videoLogID = logID

    def configure(self, args):
        self._args = args

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render_step_helper(self):
        """Helper to configure the visualizer camera during step()."""
        # Sleep, otherwise the computation takes less time than real time,
        # which will make the visualization like a fast-forward video.
        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        # time_to_sleep = self._action_repeat * self._time_step - time_spent
        time_to_sleep = self._time_step - time_spent
        if time_to_sleep > 0 and (time_to_sleep < self._time_step):
            time.sleep(time_to_sleep)

        base_pos = self.robot.GetBasePosition()
        camInfo = self._pybullet_client.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]
        targetPos = [0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1], curTargetPos[2]]
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)

    def _configure_visualizer(self):
        """Remove all visualizer borders, and zoom in"""
        # default rendering options
        self._render_width = 333
        self._render_height = 480
        self._cam_dist = 1.5
        self._cam_yaw = 20
        self._cam_pitch = -20
        # get rid of visualizer things
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def addLine(self, lineFromXYZ, lineToXYZ, lifeTime=0, color=[1, 0, 0]):
        """Add line between point A and B for duration lifeTime"""
        self._pybullet_client.addUserDebugLine(lineFromXYZ, lineToXYZ, lineColorRGB=color, lifeTime=lifeTime)

    def get_sim_time(self):
        """Get current simulation time."""
        return self._sim_step_counter * self._time_step

    def scale_rand(self, num_rand, low, high):
        """scale number of rand numbers between low and high"""
        return low + np.random.random(num_rand) * (high - low)

    def add_random_boxes(self, num_rand=100, z_height=0.04):
        """Add random boxes in front of the robot in x [0.5, 20] and y [-3,3]"""
        # x location
        x_low, x_upp = 0.5, 20
        # y location
        y_low, y_upp = -3, 3
        # block dimensions
        block_x_min, block_x_max = 0.1, 1
        block_y_min, block_y_max = 0.1, 1
        z_low, z_upp = 0.005, z_height
        # block orientations
        roll_low, roll_upp = -0.01, 0.01
        pitch_low, pitch_upp = -0.01, 0.01
        yaw_low, yaw_upp = -np.pi, np.pi

        x = x_low + np.random.random(num_rand) * (x_upp - x_low)
        y = y_low + np.random.random(num_rand) * (y_upp - y_low)
        z = z_low + np.random.random(num_rand) * (z_upp - z_low)
        block_x = self.scale_rand(num_rand, block_x_min, block_x_max)
        block_y = self.scale_rand(num_rand, block_y_min, block_y_max)
        roll = self.scale_rand(num_rand, roll_low, roll_upp)
        pitch = self.scale_rand(num_rand, pitch_low, pitch_upp)
        yaw = self.scale_rand(num_rand, yaw_low, yaw_upp)
        # loop through
        for i in range(num_rand):
            sh_colBox = self._pybullet_client.createCollisionShape(
                self._pybullet_client.GEOM_BOX, halfExtents=[block_x[i] / 2, block_y[i] / 2, z[i] / 2]
            )
            orn = self._pybullet_client.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
            block2 = self._pybullet_client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=sh_colBox, basePosition=[x[i], y[i], z[i] / 2], baseOrientation=orn
            )
            # set friction coeff
            self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

        # add walls
        orn = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
        sh_colBox = self._pybullet_client.createCollisionShape(
            self._pybullet_client.GEOM_BOX, halfExtents=[x_upp / 2, 0.5, 0.5]
        )
        block2 = self._pybullet_client.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=sh_colBox, basePosition=[x_upp / 2, y_low, 0.5], baseOrientation=orn
        )
        block2 = self._pybullet_client.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=sh_colBox, basePosition=[x_upp / 2, -y_low, 0.5], baseOrientation=orn
        )

    def _add_base_mass_offset(self, spec_mass=None, spec_location=None):
        """Attach mass to robot base."""
        quad_base = np.array(self.robot.GetBasePosition())
        quad_ID = self.robot.quadruped

        offset_low = np.array([-0.15, -0.05, -0.05])
        offset_upp = np.array([0.15, 0.05, 0.05])
        if spec_location is None:
            block_pos_delta_base_frame = self.scale_rand(3, offset_low, offset_upp)
        else:
            block_pos_delta_base_frame = np.array(spec_location)
        if spec_mass is None:
            base_mass = 8 * np.random.random()
        else:
            base_mass = spec_mass
        if self._is_render:
            print("=========================== Random Mass:")
            print("Mass:", base_mass, "location:", block_pos_delta_base_frame)
            # if rendering, also want to set the halfExtents accordingly
            # 1 kg water is 0.001 cubic meters
            boxSizeHalf = [(base_mass * 0.001) ** (1 / 3) / 2] * 3
            translationalOffset = [0, 0, 0.1]
        else:
            boxSizeHalf = [0.05] * 3
            translationalOffset = [0] * 3

        sh_colBox = self._pybullet_client.createCollisionShape(
            self._pybullet_client.GEOM_BOX, halfExtents=boxSizeHalf, collisionFramePosition=translationalOffset
        )
        base_block_ID = self._pybullet_client.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=sh_colBox,
            basePosition=quad_base + block_pos_delta_base_frame,
            baseOrientation=[0, 0, 0, 1],
        )

        cid = self._pybullet_client.createConstraint(
            quad_ID,
            -1,
            base_block_ID,
            -1,
            self._pybullet_client.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            -block_pos_delta_base_frame,
        )
        # disable self collision between box and each link
        for i in range(-1, self._pybullet_client.getNumJoints(quad_ID)):
            self._pybullet_client.setCollisionFilterPair(quad_ID, base_block_ID, i, -1, 0)


def test_env():
    env = QuadrupedGymEnv(
        robot_model="GO1",
        render=True,
        on_rack=False,
        motor_control_mode="PD",
        action_repeat=10,
        enable_springs=False,
        add_noise=False,
        enable_action_interpolation=False,
        enable_action_filter=False,
        enable_action_clipping=False,
        task_env="JUMPING_ON_PLACE_TASK",
        observation_space_mode="JUMPING_ON_PLACE_OBS",
    )
    sim_steps = 1000

    obs = env.reset()
    for i in range(sim_steps):
        action = np.random.rand(12) * 2 - 1
        # action = np.full(12,0)
        obs, reward, done, info = env.step(action)
    print("end")


if __name__ == "__main__":
    # test out some functionalities
    test_env()
    os.sys.exit()

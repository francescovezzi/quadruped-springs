"""This file implements the gym environment for a quadruped. """
import datetime
import os
import time

# gym
import gym
import numpy as np

# pybullet
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
from gym import spaces
from gym.utils import seeding

import quadruped_spring.go1.configs_go1_with_springs as go1_config_with_springs
import quadruped_spring.go1.configs_go1_without_springs as go1_config_without_springs
from quadruped_spring.env import quadruped
from quadruped_spring.env.control_interface.collection import ActionInterfaceCollection, MotorInterfaceCollection
from quadruped_spring.env.control_interface.utils import settle_robot_by_pd
from quadruped_spring.env.env_randomizers.env_randomizer_collection import EnvRandomizerCollection
from quadruped_spring.env.env_randomizers.env_randomizer_list import EnvRandomizerList
from quadruped_spring.env.sensors.robot_sensors import SensorList
from quadruped_spring.env.sensors.sensor_collection import SensorCollection
from quadruped_spring.env.tasks.task_collection import TaskCollection
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.utils import action_filter

# from quadruped_spring.env.wrappers.rest_wrapper import RestWrapper
# from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
EPISODE_LENGTH = 10  # max episode length for RL (seconds)
VIDEO_LOG_DIRECTORY = "videos/" + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")


# Motor control mode implemented: TORQUE, PD, CARTESIAN_PD
# Observation space implemented: DEFAULT, ENCODER, CARTESIAN_NO_IMU, ANGLE_NO_IMU
# Action space implemented: DEFAULT, SYMMETRIC, SYMMETRIC_NO_HIP
# Task implemented: JUMPING_ON_PLACE_HEIGHT, JUMPING_FORWARD
# Env randomizer implemented: MASS_RANDOMIZER, DISTURBANCE_RANDOMIZER, SETTLING_RANDOMIZER, SPRING_RANDOMIZER

# NOTE:
# TORQUE control mode actually works only if isRLGymInterface is setted to False.


class QuadrupedGymEnv(gym.Env):
    """
    The gym environment for a quadruped {Unitree GO1}.

    It simulates the locomotion of a quadrupedal robot.
    The state space, action space, and reward functions can be chosen with:
    observation_space_mode, motor_control_mode, task_env.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        isRLGymInterface=True,
        time_step=0.001,
        action_repeat=10,
        motor_control_mode="PD",
        task_env="JUMPING_ON_PLACE_HEIGHT",
        observation_space_mode="DEFAULT",
        action_space_mode="DEFAULT",
        on_rack=False,
        render=False,
        record_video=False,
        add_noise=True,
        enable_springs=False,
        enable_action_interpolation=False,
        enable_action_filter=False,
        enable_env_randomization=True,
        env_randomizer_mode="MASS_RANDOMIZER",
        curriculum_level=0.0,
        test_env=False,  # NOT ALLOWED FOR TRAINING!
    ):
        """Initialize the quadruped gym environment.

        Args:
          isRLGymInterface: If the gym environment is being run as RL or not. Affects
            if the actions should be scaled.
          time_step: Simulation time step.
          action_repeat: The number of simulation steps where the same actions are applied.
          distance_weight: The weight of the distance term in the reward.
          energy_weight: The weight of the energy term in the reward.
          motor_control_mode: Whether to use torque control, PD, control, etc.
          task_env: Task trying to learn (fwd locomotion, standup, etc.)
          observation_space_mode: what should be in here? Check available functions in quadruped.py
          action_space_mode: For action space dimension selecting
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
          enable_joint_velocity_estimate: Boolean specifying if it's used the
            estimated or the true joint velocity. Actually it affects only real
            observations space modes.
          enable_env_randomizer: Boolean specifying whether to enable env randomization.
          env_randomizer_mode: String specifying which env randomizers to use.
          curriculum_level: Scalar in [0,1] specyfing the task difficulty level.
        """
        self.seed()
        self._enable_springs = enable_springs
        if self._enable_springs:
            self._robot_config = go1_config_with_springs
        else:
            self._robot_config = go1_config_without_springs
        self._isRLGymInterface = isRLGymInterface
        self._sim_time_step = time_step
        self._action_repeat = action_repeat
        self._env_time_step = self._action_repeat * self._sim_time_step
        self._hard_reset = True  # must fully reset simulation at init
        self._on_rack = on_rack
        self._is_render = render
        self._is_record_video = record_video
        self._add_noise = add_noise
        self._enable_action_interpolation = enable_action_interpolation
        self._enable_action_filter = enable_action_filter
        self._using_test_env = test_env
        if test_env:
            self._add_noise = True

        # other bookkeeping
        self._num_bullet_solver_iterations = int(300 / action_repeat)
        self._last_frame_time = 0.0  # for rendering
        self._MAX_EP_LEN = EPISODE_LENGTH  # max sim time in seconds, arbitrary
        self._action_bound = 1.0

        self._build_action_command_interface(motor_control_mode, action_space_mode)
        self.setupActionSpace()

        self._observation_space_mode = observation_space_mode
        self._robot_sensors = SensorList(SensorCollection().get_el(self._observation_space_mode))
        self.setupObservationSpace()

        self.task_env = task_env
        self.task = TaskCollection().get_el(self.task_env)()
        self.task.set_curriculum_level(curriculum_level, verbose=0)

        if self._enable_action_filter:
            self._action_filter = self._build_action_filter()

        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()

        self.videoLogID = None
        self._enable_env_randomization = enable_env_randomization
        if self._enable_env_randomization:
            self._env_randomizer_mode = env_randomizer_mode
            self._env_randomizers = EnvRandomizerList(EnvRandomizerCollection().get_el(self._env_randomizer_mode))
            self._env_randomizers._init(self)

        self.reset()

    ######################################################################################
    # RL Observation and Action spaces
    ######################################################################################
    def setupObservationSpace(self):
        self._robot_sensors._init(robot_config=self._robot_config)
        obs_high = self._robot_sensors._get_high_limits() + OBSERVATION_EPS
        obs_low = self._robot_sensors._get_low_limits() - OBSERVATION_EPS
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def _build_action_command_interface(self, motor_control_mode, action_space_mode):
        if motor_control_mode == "TORQUE" and self._isRLGymInterface:
            raise ValueError(f"the motor control mode {motor_control_mode} not" "implemented yet for RL Gym interface.")

        motor_interface = MotorInterfaceCollection().get_el(motor_control_mode)
        motor_interface = motor_interface(self)

        ac_interface = ActionInterfaceCollection().get_el(action_space_mode)
        self._ac_interface = ac_interface(motor_interface)

        self._motor_control_mode = motor_control_mode
        self._action_space_mode = action_space_mode

    def setupActionSpace(self):
        """Set up action space for RL."""
        self._action_dim = self.get_action_dim()
        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self._last_action = np.zeros(self._action_dim)

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

    def _sub_step(self, action, sub_step):
        if self._isRLGymInterface:
            if self._enable_action_interpolation:
                action = self._interpolate_actions(action, sub_step)
            proc_action = self._ac_interface._transform_action_to_motor_command(action)
        else:
            proc_action = action
        self.robot.ApplyAction(proc_action)
        if self._enable_env_randomization:
            self._env_randomizers.randomize_step()

        self._pybullet_client.stepSimulation()
        self._sim_step_counter += 1

        if self._is_render:
            self._render_step_helper()

    def step(self, action):
        """Step forward the simulation, given the action."""
        curr_act = action.copy()
        self._last_action = curr_act

        if self._enable_action_filter:
            curr_act = self._filter_action(curr_act)

        for sub_step in range(self._action_repeat):
            self._sub_step(curr_act, sub_step)

        self._env_step_counter += 1
        self.task._on_step()
        reward = self.task._reward()
        done = False
        # infos = {"base_pos": self.robot.GetBasePosition()}
        infos = {}

        task_terminated = self.task_terminated()
        if task_terminated or self.get_sim_time() > self._MAX_EP_LEN:
            infos["TimeLimit.truncated"] = not task_terminated
            done = True

        # Update the actual reward at the end of the episode with bonus or malus
        if done:
            reward += self.task._reward_end_episode()

        self._robot_sensors._on_step()
        obs = self.get_observation()

        return obs, reward, done, infos

    ###################################################
    # Filtering to smooth actions
    ###################################################
    def _build_action_filter(self):
        sampling_rate = 1 / self._env_time_step
        num_joints = self._action_dim
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
        # if self._enable_springs:
        #     a_filter.highcut = 2.5
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
        init_action = self._last_action
        self._action_filter.init_history(init_action)

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
            self._pybullet_client.setTimeStep(self._sim_time_step)
            self.plane = self._pybullet_client.loadURDF(
                pybullet_data.getDataPath() + "/plane.urdf", basePosition=[80, 0, 0]
            )  # to extend available running space (shift)
            self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._pybullet_client.setGravity(0, 0, -9.8)
            self.robot = quadruped.Quadruped(
                pybullet_client=self._pybullet_client,
                robot_config=self._robot_config,
                motor_control_mode=self._ac_interface._motor_control_mode_ROB,
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

        if self._is_render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])

        self._ac_interface._reset(self.robot)
        if self._enable_env_randomization:
            self._env_randomizers.randomize_env()
        self._settle_robot()  # Settle robot after being spawned
        self.task._reset(self)  # Reset task internal state
        self._robot_sensors._reset(self.robot)  # Rsest sensors

        if self._enable_action_filter:
            self._reset_action_filter()
            self._init_filter()

        if self._is_record_video:
            self.recordVideoHelper()

        return self.get_observation()

    def _settle_robot(self):
        if self._isRLGymInterface:
            self._last_action = self._ac_interface._settle_robot_by_reference(self.get_init_pose(), n_steps=1200)
        else:
            settle_robot_by_pd(self)

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
        time_to_sleep = self._sim_time_step - time_spent
        if time_to_sleep > 0 and (time_to_sleep < self._sim_time_step):
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

    ########################################################
    # Get methods
    ########################################################
    def get_action_dim(self):
        return self._ac_interface.get_action_space_dim()

    def get_observation(self):
        return self._robot_sensors.get_noisy_obs()

    def get_sim_time(self):
        """Get current simulation time."""
        return self._sim_step_counter * self._sim_time_step

    def get_env_time_step(self):
        """Get environment simulation time step."""
        return self._env_time_step

    def get_motor_control_mode(self):
        """Get current motor control mode."""
        return self._motor_control_mode

    def get_robot_config(self):
        """Get current robot config."""
        return self._robot_config

    def are_springs_enabled(self):
        """Get boolean specifying if springs are enabled or not."""
        return self._enable_springs
    
    def get_action_observation_space_mode(self):
        """Get action and observation space mode."""
        return self._action_space_mode, self._observation_space_mode
    
    def low_pass_filter_enabled(self):
        """Get boolean specifying if low-pass filter is enabled."""
        return self._enable_action_filter

    def task_terminated(self):
        """Return boolean specifying whther the task is terminated."""
        return self.task._terminated()

    def get_reward_end_episode(self):
        """Return bonus and malus to add to the reward at the end of the episode."""
        return self.task._reward_end_episode()

    def get_init_pose(self):
        """Get the initial init pose for robot settling."""
        return self._ac_interface.get_init_pose()

    def get_settling_action(self):
        """Get the settling action."""
        return self._settling_action

    def get_landing_action(self):
        """Get the action the landing controller should apply."""
        landing_pose = self._ac_interface.get_landing_pose()
        landing_action = self._ac_interface._transform_motor_command_to_action(landing_pose)
        return landing_action

    def get_last_action(self):
        """Get the last action applied."""
        return self._last_action
    
    def print_task_info(self):
        """Print some info about the task performed."""
        self.task.print_info()


def build_env():
    env_config = {
        "render": True,
        "on_rack": False,
        "motor_control_mode": "PD",
        "action_repeat": 10,
        "enable_springs": False,
        "add_noise": False,
        "enable_action_interpolation": False,
        "enable_action_filter": True,
        "task_env": "JUMPING_ON_PLACE_HEIGHT",
        "observation_space_mode": "ARS_HEIGHT",
        "action_space_mode": "SYMMETRIC",
        "enable_env_randomization": False,
        "env_randomizer_mode": "SETTLING_RANDOMIZER",
    }
    env = QuadrupedGymEnv(**env_config)
    
    env = ObsFlatteningWrapper(env)
    # env = RestWrapper(env)
    # env = LandingWrapper(env)
    return env


def test_env():
    env = build_env()
    sim_steps = 500
    action_dim = env.get_action_dim()
    obs = env.reset()
    for i in range(sim_steps):
        action = np.random.rand(action_dim) * 2 - 1
        # action = np.full(action_dim, 0)
        # action = env.get_settling_action()
        obs, reward, done, info = env.step(action)
    # env.print_task_info()
    env.close()
    print("end")


if __name__ == "__main__":
    # test out some functionalities
    test_env()
    os.sys.exit()

"""This file implements the gym environment for a quadruped. """
import datetime
import os

# gym
import gym
import numpy as np

# pybullet
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
from gym import spaces

import quadruped_spring.go1.configs_go1_with_springs as go1_config_with_springs
import quadruped_spring.go1.configs_go1_without_springs as go1_config_without_springs
from quadruped_spring.env import quadruped
from quadruped_spring.env.control_interface.collection import ActionInterfaceCollection, MotorInterfaceCollection
from quadruped_spring.env.control_interface.utils import settle_robot_by_pd
from quadruped_spring.env.env_randomizers.env_randomizer_base import EnvRandomizerList
from quadruped_spring.env.env_randomizers.env_randomizer_collection import EnvRandomizerCollection
from quadruped_spring.env.sensors.sensor import SensorList
from quadruped_spring.env.sensors.sensor_collection import SensorCollection
from quadruped_spring.env.tasks.task_collection import TaskCollection
from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper
from quadruped_spring.env.wrappers.reference_state_initialization_wrapper import ReferenceStateInitializationWrapper
from quadruped_spring.env.wrappers.landing_wrapper_backflip import LandingWrapperBackflip
from quadruped_spring.env.wrappers.landing_wrapper_continuous2 import LandingWrapperContinuous2
from quadruped_spring.utils import action_filter
from quadruped_spring.utils.camera import make_camera
from quadruped_spring.scripts.evaluation_wrapper import EvaluationWrapper

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
EPISODE_LENGTH = 10  # max episode length for RL (seconds)

# NOTE:
# Motor control mode implemented: TORQUE, PD, CARTESIAN_PD
# TORQUE control mode actually works only if isRLGymInterface is set to False.

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
        task_env="NO_TASK",
        observation_space_mode="ENCODER",
        action_space_mode="SYMMETRIC",
        on_rack=False,
        render=False,
        enable_springs=False,
        enable_action_interpolation=False,
        enable_action_filter=False,
        env_randomizer_mode="GROUND_RANDOMIZER",
        camera_mode="CLASSIC",
        curriculum_level=0.0,
        verbose=0,
    ):
        """Initialize the quadruped gym environment.

        Args:
          isRLGymInterface: If the gym environment is being run as RL or not. Affects
            if the actions should be scaled.
          time_step: Simulation time step.
          action_repeat: The number of simulation steps where the same actions are applied.
          motor_control_mode: Whether to use torque control, PD, control, etc.
          task_env: Task trying to learn (fwd locomotion, standup, etc.)
          observation_space_mode: what should be in here? Check available functions in quadruped.py
          action_space_mode: For action space dimension selecting
          on_rack: Whether to place the quadruped on rack. This is only used to debug
            the walking gait. In this mode, the quadruped's base is hanged midair so
            that its walking gait is clearer to visualize.
          render: Whether to render the simulation.
          enable_springs: Whether to enable springs or not
          enable_action_interpolation: Whether to interpolate the current action
            with the previous action in order to produce smoother motions
          enable_action_filter: Boolean specifying if a lowpass filter should be
            used to smooth actions.
          enable_action_clipping: Boolean specifying if motor commands should be
            clipped or not. It's not implemented for pure torque control.
          env_randomizer_mode: String specifying which env randomizers to use.
          camera_mode: Decide how to visualize the simulation.
          curriculum_level: Scalar in [0,1] specyfing the task difficulty level.
        """
        self.verbose = verbose
        self._enable_springs = enable_springs
        if self._enable_springs:
            self._robot_config = go1_config_with_springs
        else:
            self._robot_config = go1_config_without_springs
        self._isRLGymInterface = isRLGymInterface
        self.sim_time_step = time_step
        self._action_repeat = action_repeat
        self.env_time_step = self._action_repeat * self.sim_time_step
        self._on_rack = on_rack
        self._is_render = render
        self._enable_action_interpolation = enable_action_interpolation
        self._enable_action_filter = enable_action_filter

        # other bookkeeping
        self._num_bullet_solver_iterations = int(300 / action_repeat)
        self._MAX_EP_LEN = EPISODE_LENGTH  # max sim time in seconds, arbitrary
        self._settling_steps = 2500

        self.task_env = task_env

        self._build_action_command_interface(motor_control_mode, action_space_mode)
        self.action_dim = self._ac_interface.get_action_space_dim()
        self.setupActionSpace(self.action_dim)

        self.task = TaskCollection().get_el(self.task_env)(self)

        self._observation_space_mode = observation_space_mode
        self._robot_sensors = SensorList(SensorCollection().get_el(self._observation_space_mode), self)
        self.setupObservationSpace()

        if self._enable_action_filter:
            self._action_filter = self._build_action_filter()

        if self._is_render:
            opts = "--background_color_red=1 --background_color_blue=1 --background_color_green=1"
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI, options=opts)
            # self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()

        self.camera = make_camera(self, mode=camera_mode)

        self.robot_desired_state = None  # Reset the robot in a desired state
        self.reset_pybullet_simulation()

        self._env_randomizer_mode = env_randomizer_mode
        self._env_randomizers = EnvRandomizerList(EnvRandomizerCollection().get_el(self._env_randomizer_mode))
        self._env_randomizers._init(self)
        if self._env_randomizers.is_curriculum_enabled():
            self.curriculum_level = 0.0
            self.task.change_parameters()
            self._env_randomizers.increase_curriculum_level(curriculum_level)

        self.sub_step_callback = None  # For doing stuff at sim_step frequency

        if self.verbose > 0:
            self.print_info()

    ######################################################################################
    # RL Observation and Action spaces
    ######################################################################################
    def setupObservationSpace(self):
        self._robot_sensors._init(self.get_robot_config())
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

    def setupActionSpace(self, action_dim):
        """Set up action space for RL."""
        action_high = np.array([1] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

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
        last_action = self._last_action if not self._enable_action_filter else self._last_filtered_action
        if self._enable_action_interpolation and self._last_action is not None:
            interp_fraction = float(substep_count + 1) / self._action_repeat
            interpolated_action = last_action + interp_fraction * (action - last_action)
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
        # self._env_randomizers.randomize_step()
        self.step_simulation()

    def step_simulation(self, increase_sim_counter=True):
        self._pybullet_client.stepSimulation()
        if increase_sim_counter:
            self._sim_step_counter += 1
            if self.sub_step_callback is not None:
                self.sub_step_callback()
        if self._is_render:
            self.camera._render_step_helper()

    def step(self, action):
        """Step forward the simulation, given the action."""
        curr_act = action.copy()
        self._last_action = curr_act

        if self._enable_action_filter:
            curr_act = self._filter_action(curr_act)
            self._last_filtered_action = curr_act

        for sub_step in range(self._action_repeat):
            self._sub_step(curr_act, sub_step)

        self._env_step_counter += 1
        self.task._on_step()
        reward = self.task._reward()
        done = False
        infos = {}
        task_terminated = self.task._terminated()
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
        sampling_rate = 1 / self.env_time_step
        num_joints = self.action_dim
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
        return a_filter

    def _reset_action_filter(self):
        self._action_filter.reset()
        self._action_filter.init_history(self._last_action)

    def _filter_action(self, action):
        filtered_action = self._action_filter.filter(action)
        return filtered_action

    ######################################################################################
    # Reset
    ######################################################################################
    def reset(self):
        """Set up simulation environment."""
        self.reset_pybullet_simulation()

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_action = self._last_filtered_action = np.zeros(self.action_dim)
        self._ac_interface._reset(self.robot)
        self._env_randomizers.randomize_env()

        if self.robot_desired_state is None:
            self._settle_robot()  # Settle robot after being spawned
        self.task._reset()  # Reset task internal state
        self._env_randomizers.randomize_robot()
        self._robot_sensors._reset(self.robot)  # Rsest sensors

        if self._enable_action_filter:
            self._reset_action_filter()

        return self.get_observation()

    def reset_pybullet_simulation(self):
        # set up pybullet simulation
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self.sim_time_step)
        self.plane = self._pybullet_client.loadURDF(
            pybullet_data.getDataPath() + "/plane.urdf", basePosition=[80, 0, 0]
        )  # to extend available running space (shift)
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._quadruped_config = dict(
            pybullet_client=self._pybullet_client,
            robot_config=self._robot_config,
            motor_control_mode=self._ac_interface._motor_control_mode_ROB,
            on_rack=self._on_rack,
            render=self._is_render,
            enable_springs=self._enable_springs,
            desired_state=self.robot_desired_state,
        )
        self.robot = quadruped.Quadruped(**self._quadruped_config)
        if self._is_render:
            self.camera.reset()

    def _settle_robot(self):
        if self._isRLGymInterface:
            self._last_action = self._ac_interface._settle_robot_by_reference(
                self.get_init_pose(), n_steps=self._settling_steps
            )
        else:
            settle_robot_by_pd(self)

    ######################################################################################
    # Render and Close
    ######################################################################################
    def render(self, mode="rgb_array"):
        return self.camera.render(mode)

    def close(self):
        self._pybullet_client.disconnect()

    ########################################################
    # Get methods
    ########################################################
    def get_observation(self):
        """Return observation."""
        return self._robot_sensors.get_noisy_obs()

    def get_sim_time(self):
        """Get current simulation time."""
        return self._sim_step_counter * self.sim_time_step

    def get_motor_control_mode(self):
        """Get current motor control mode."""
        return self._motor_control_mode

    def get_robot_config(self):
        """Get current robot config."""
        return self._robot_config

    def are_springs_enabled(self):
        """Return boolean specifying whether springs are enabled."""
        return self._enable_springs

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

    def get_last_filtered_action(self):
        """Get the last filtered action."""
        return self._last_filtered_action

    def get_observation_space_mode(self):
        """Get the observation space mode."""
        return self._observation_space_mode

    def get_curriculum_level(self):
        """Return the acutal curriculum level."""
        return self.curriculum_level

    def get_quadruped_config(self):
        """Return the quadruped configuration."""
        return self._quadruped_config

    def set_robot_desired_state(self, state):
        self.robot_desired_state = state

    def set_sub_step_callback(self, callback):
        self.sub_step_callback = callback

    def get_randomizer_mode(self):
        """Return the env ranodmizer mode."""
        return self._env_randomizer_mode

    def reinit_randomizers(self, env):
        """Reinitialize randomizers for wrapped env."""
        self._env_randomizers._reinit(env)

    def reinit_sensors(self, env):
        """Reinitialize sensors for wrapped env."""
        self._robot_sensors._reinit(env)

    def get_ac_interface(self):
        """Return the action control interface."""
        return self._ac_interface

    def increase_curriculum_level(self, value):
        """increase the curriculum level."""
        assert value >= 0 and value <= 1, "curriculum level change should be in [0,1]."
        self._env_randomizers.increase_curriculum_level(value)

    def print_info(self):
        """Print environment info."""
        print("\n*** Environment Info ***")
        print(f"task environment -> {self.task_env}")
        print(f"spring enabled -> {self._enable_springs}")
        print(f"low-pass action filter > {self._enable_action_filter}")
        print(f"sensors -> {self._observation_space_mode}")
        print(f"env randomizer -> {self._env_randomizer_mode}")
        print("")


def build_env():
    env_config = {
        "render": False,
        "on_rack": False,
        "motor_control_mode": "PD",
        "action_repeat": 10,
        "enable_springs": True,
        "enable_action_interpolation": False,
        "enable_action_filter": True,
        "task_env": "CONTINUOUS_JUMPING_FORWARD_DEMO",
        "observation_space_mode": "PPO_CONTINUOUS_JUMPING_FORWARD",
        "action_space_mode": "SYMMETRIC",
        "env_randomizer_mode": "GROUND_RANDOMIZER",
        "camera_mode": "CLASSIC",
        "curriculum_level": 1.0,
    }
    env = QuadrupedGymEnv(**env_config)
    # env = ReferenceStateInitializationWrapper(env)
    env = EvaluationWrapper(env)
    return env


def test_env():
    env = build_env()
    sim_steps = 1500
    obs = env.reset()
    done = False
    action_dim = env.action_dim
    rew = 0

    for i in range(sim_steps):
        action = np.random.rand(action_dim) * 2 - 1
        obs, reward, done, info = env.step(action)
        rew += reward
        if done:
            break
    print(f"rew: {rew}")
    env.close()
    print("end")


if __name__ == "__main__":
    # test out some functionalities
    test_env()
    os.sys.exit()

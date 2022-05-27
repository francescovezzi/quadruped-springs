import time

import numpy as np


class MotorInterfaceBase:
    """Prototype class for a generic command-action interface"""

    def __init__(self, env):
        self._robot = None
        self._motor_control_mode = None
        self._motor_control_mode_ROB = None
        self._lower_lim = None
        self._upper_lim = None
        self._init_pose = None
        self._settling_pose = None
        self._landing_pose = None
        self._action_bound = 1.0
        self._init(env)

    def _init(self, env):
        """Initialize the interface"""
        self._env = env
        self._robot_config = self._env._robot_config

    def _reset(self, robot):
        """Reset interface"""
        self._robot = robot

    def set_init_pose(self, init_pose):
        """Set the robot initial pose."""
        assert len(init_pose) == self._robot_config.NUM_MOTORS, "Wrong dimension for init pose."
        self._init_pose = init_pose

    def set_settling_pose(self, settle_pose):
        self._settling_pose = settle_pose

    def set_landing_pose(self, land_pose):
        self._landing_pose = land_pose

    def get_init_pose(self):
        """Get the initial pose robot should be settled at reset."""
        return self._init_pose

    def get_landing_pose(self):
        """Get the pose you'd like the robot assume at landing."""
        return self._landing_pose

    def get_settling_pose(self):
        """Get the settling pose you want the robot to achieve."""
        return self._settling_pose

    def get_motor_control_mode(self):
        """Get the implemented motor control mode."""
        return self._motor_control_mode

    def get_robot_pose(self):
        """Get the robot pose."""
        pass

    def _transform_action_to_motor_command(self, action):
        """Convert the action to the properly motor command."""
        pass

    def _transform_motor_command_to_action(self, command):
        """Convert the motor command to the action that generates it."""
        pass

    def _convert_reference_to_command(self, reference):
        """To generate the right robot input based on the command reference."""
        action_ref = self._transform_motor_command_to_action(reference)
        command_ref = self._transform_action_to_motor_command(action_ref)
        return command_ref

    def get_init_action(self):
        """The action that push the robot toward the init pose"""
        return self._transform_motor_command_to_action(self.get_init_pose())

    def get_landing_action(self):
        """The action that push the robot toward the landing pose"""
        return self._transform_motor_command_to_action(self.get_landing_pose())

    def _scale_helper_action_to_motor_command(self, action):
        """Helper to linearly scale from [-1,1] to lower/upper limits."""
        action = np.clip(action, -1, 1)
        lower_lim = self._lower_lim
        upper_lim = self._upper_lim
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        return np.clip(new_a, lower_lim, upper_lim)

    def _scale_helper_motor_command_to_action(self, command):
        """
        Helper to linearly scale from [lower_lim, upper_lim] to [-1, 1].
        """
        lower_lim = self._lower_lim
        upper_lim = self._upper_lim
        command = np.clip(command, lower_lim, upper_lim)
        action = -1 + 2 * (command - lower_lim) / (upper_lim - lower_lim)
        return np.clip(action, -1, 1)

    def get_intermediate_settling_pose(self, i):
        """
        Get a pose that is the result of an interpolation between
        the initial pose and the settling pose. The parameter i belongs to [0, 1].
        """
        assert i >= 0 and i <= 1, "the interpolation parameter should belongs to [0, 1]."
        init_pose = self.get_init_pose()
        end_pose = self.get_settling_pose()
        return init_pose * (1 - i) + i * end_pose

    @staticmethod
    def generate_ramp(i, i_min, i_max, u_min, u_max) -> float:
        """Return the output from a ramp function."""
        if i < i_min:
            return u_min
        elif i > i_max:
            return u_max
        else:
            return u_min + (u_max - u_min) * (i - i_min) / (i_max - i_min)

    def smooth_settling(self, i, i_min, i_max, j=1):
        """Return the output from a ramp going from the init pose to the settling pose."""
        return self.generate_ramp(i, i_min, i_max, self.get_init_pose(), self.get_intermediate_settling_pose(j))


MOTOR_CONTROL_MODE_SUPPORTED_LIST = ["TORQUE", "PD", "CARTESIAN_PD"]


class ActionWrapperBase(MotorInterfaceBase):
    """Wrapper for taking into account the action space."""

    def __init__(self, motor_interface):
        self._motor_interface = motor_interface
        self._action_space_mode = "please give me a name"
        self._action_dim = None
        if self._motor_interface._motor_control_mode not in MOTOR_CONTROL_MODE_SUPPORTED_LIST:
            raise ValueError(
                f"the motor control mode {self._motor_interface._motor_control_mode} is not "
                f"implemented yet for gym interface."
            )

    def __getattr__(self, name):
        return getattr(self._motor_interface, name)

    def _reset(self, robot):
        self._motor_interface._reset(robot)

    def _convert_to_default_action_space(self, action):
        """
        Transform the action in the proper action of dimension 12 (NUM_LEGS)
        before being processed to become the motor command.
        """
        pass

    def _convert_to_actual_action_space(self, action):
        """
        Transform the action in the proper action of dimension _action_dim
        to help conversion from command to action.
        """
        pass

    def _transform_action_to_motor_command(self, action):
        action = self._convert_to_default_action_space(action)
        return self._motor_interface._transform_action_to_motor_command(action)

    def _transform_motor_command_to_action(self, command):
        action = self._motor_interface._transform_motor_command_to_action(command)
        return self._convert_to_actual_action_space(action)

    def get_action_space_mode(self):
        """Get the implemented action space mode."""
        return self._action_space_mode

    def get_action_space_dim(self):
        """Get the action space dimension."""
        return self._action_dim

    def get_landing_pose(self):
        return self._motor_interface.get_landing_pose()

    def get_init_pose(self):
        return self._motor_interface.get_init_pose()

    def set_init_pose(self, init_pose):
        self._motor_interface.set_init_pose(init_pose)
        
    def set_settling_pose(self, settle_pose):
        self._motor_interface.set_settling_pose(settle_pose)

    def set_landing_pose(self, land_pose):
        self._motor_interface.set_landing_pose(land_pose)

    def get_robot_pose(self):
        return self._motor_interface.get_robot_pose()

    def get_settling_pose(self):
        return self._motor_interface.get_settling_pose()

    def smooth_settling(self, i, i_min, i_max, j=1):
        return self._motor_interface.smooth_settling(i, i_min, i_max, j)

    def get_intermediate_settling_pose(self, i):
        return self._motor_interface.get_intermediate_settling_pose(i)

    def _settle_robot_by_reference(self, reference, n_steps):
        """
        Settle robot in according to the used motor control mode in RL interface.
        Return the last action utilized.
        The reference is the desired pose for the robot, can be expressed either in
        joint space (joint angles) that in cartesian space (feet position).
        The command is as the same as reference except the fact the output produced
        respect the Action Space mode selected, e.g. for Symmetric mode the output
        is forced to be symmetric in this way.
        """
        env = self._motor_interface._env
        if env._is_render:
            time.sleep(0.2)
        settling_command = self._convert_reference_to_command(reference)
        settling_action = self._transform_motor_command_to_action(settling_command)
        for _ in range(n_steps):
            env.robot.ApplyAction(settling_command)
            if env._is_render:
                time.sleep(0.001)
            env._pybullet_client.stepSimulation()
        return settling_action

    def _load_springs(self, j=0.5):
        """Settle the robot to an initial config. Return last action used."""
        env = self._motor_interface._env
        if env._is_render:
            time.sleep(0.2)
        n_steps_tot = 900
        n_steps_ramp = max(n_steps_tot - 100, 1)
        for i in range(n_steps_tot):
            reference = self.smooth_settling(i, 0, n_steps_ramp, j)
            settling_command = self._convert_reference_to_command(reference)
            env.robot.ApplyAction(settling_command)
            if env._is_render:
                time.sleep(0.001)
            env._pybullet_client.stepSimulation()
        return self._transform_motor_command_to_action(settling_command)

import numpy as np


class MotorInterfaceBase:
    """Prototype class for a generic command-action interface"""

    def __init__(self, robot_config):
        self._robot = None
        self._motor_control_mode = None
        self._motor_control_mode_ROB = None
        self._lower_lim = None
        self._upper_lim = None
        self._init_pose = None
        self._action_bound = 1.0
        self._init(robot_config)

    def _init(self, robot_config):
        """Initialize the interface"""
        self._robot_config = robot_config

    def _reset(self, robot):
        """Reset interface"""
        self._robot = robot

    def get_init_pose(self):
        """Get the initial pose robot should be settled at reset."""
        return self._init_pose

    def get_landing_pose(self):
        """Get the pose you'd like the robot assume at landing."""
        pass

    def get_motor_control_mode(self):
        """Get the implemented motor control mode."""
        return self._motor_control_mode

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
        self._motor_interface._robot = robot

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

import numpy as np


class MotorInterfaceBase():
    """Prototype class for a generic command-action interface"""
    
    def __init__(self, robot_config):
        self._robot = None
        self._motor_control_mode = None
        self._lower_lim = None
        self._upper_lim = None
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
        raise NotImplementedError
    
    def get_landing_pose(self):
        """Get the pose you'd like the robot assume at landing."""
        return self._robot_config.LANDING_POSE
    
    def _transform_action_to_motor_command(self, action):
        """Convert the action to the properly motor command."""
        command = self._scale_helper_action_to_motor_command(
                action, self.lower_lim, self.upper_lim
            )
        return command
    
    def _transform_motor_command_to_action(self, command):
        """Convert the motor command to the action that generates it."""
        action = self._scale_helper_motor_command_to_action(
            command, self.lower_lim, self.upper_lim
            )
        return action
    
    def get_init_action(self):
        """The action that push the robot toward the init pose"""
        return self._transform_motor_command_to_action(self.get_init_pose())
    
    def get_landing_action(self):
        """The action that push the robot toward the landing pose"""
        return self._transform_motor_command_to_action(self.get_landing_pose())
    
    @staticmethod
    def _scale_helper_action_to_motor_command(self, action, lower_lim, upper_lim):
        """Helper to linearly scale from [-1,1] to lower/upper limits."""
        bound = self._action_bound
        action = np.clip(action, -1, 1)
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        return np.clip(new_a, lower_lim, upper_lim)
    
    @staticmethod
    def _scale_helper_motor_command_to_action(self, command, min_command, max_command):
        """
        Helper to linearly scale from [min_command, max_command] to [-1, 1].
        """
        command = np.clip(command, min_command, max_command)
        action = -1 + 2 * (command - min_command) / (max_command - min_command)
        return np.clip(action, -1, 1)

import numpy as np

MOTOR_CONTROL_MODE_SUPPORTED_LIST = ["PD", "CARTESIAN_PD"]

class ActionWrapperBase():
    """Wrapper for taking into account the action space."""
    def __init__(self, motor_interface):
        self._motor_interface = motor_interface
        self._action_space_mode = "please give me a name"
        self._action_dim = None
        if self._motor_control_mode not in MOTOR_CONTROL_MODE_SUPPORTED_LIST:
            raise ValueError("the motor control mode {self._motor_control_mode} is not "
                             "implemented yet for gym interface.")
        
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
    

class DefaultActionWrapper(ActionWrapperBase):
    """Action Wrapper for default space."""
    def __init__(self, motor_interface):
        super().__init__(motor_interface)
        self._action_space_mode = "DEFAULT"
        self._action_dim = 12
        
    def _convert_to_default_action_space(self, action):
        return action
        
    def _convert_to_default_action_space(self, action):
        return action
        
class SymmetricActionWrapper(ActionWrapperBase):
    """Action Wrapper to assign symmetric action for the robot right and left side."""
    def __init__(self, motor_interface):
        super().__init__(motor_interface)
        self._action_space_mode = "SYMMETRIC"
        self._action_dim = 6
      
    def _convert_to_default_action_space(self, action):
        leg_FR = action[0:3]
        leg_RR = action[3:6]

        leg_FL = leg_FR
        leg_FL[self._symm_idx] = -leg_FR[self._symm_idx]

        leg_RL = leg_RR
        leg_RL[self._symm_idx] = -leg_RR[self._symm_idx]

        return np.concatenate((leg_FR, leg_FL, leg_RR, leg_RL))
    
    def _convert_to_actual_action_space(self, action):
        leg_FR = action[0:3]
        leg_RR = action[6:9]
        return np.concatenate((leg_FR, leg_RR))
    
class SymmetricNoHipActionWrapper(ActionWrapperBase):
    """
    Action Wrapper to assign symmetric action for the robot right and left side
    plus forcing the hip action to be 0.
    """
    def __init__(self, motor_interface):
        super().__init__(motor_interface)
        self._action_space_mode = "SYMMETRIC_NO_HIP"
        self._action_dim = 4

    def _convert_to_default_action_space(self, action):
        leg_FR = action[0:2]
        leg_RR = action[2:4]

        leg_FL = leg_FR = np.insert(leg_FR, self._symm_idx, 0)
        leg_RL = leg_RR = np.insert(leg_RR, self._symm_idx, 0)

        return np.concatenate((leg_FR, leg_FL, leg_RR, leg_RL))

    def _convert_to_actual_action_space(self, action):
        leg_FR = action[0:3]
        leg_RR = action[6:9]
        
        leg_FR = np.delete(leg_FR, self._symm_idx)
        leg_RR = np.delete(leg_RR, self._symm_idx)
        
        return np.concatenate((leg_FR, leg_RR))
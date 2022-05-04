import numpy as np

from quadruped_spring.env.control_interface.interface_base import ActionWrapperBase


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

        leg_FL = np.copy(leg_FR)
        leg_FL[self._symm_idx] = -leg_FR[self._symm_idx]
        # print(self._symm_idx)
        leg_RL = np.copy(leg_RR)
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

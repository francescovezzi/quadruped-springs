from quadruped_spring.env.control_interface import action_interface as ai
from quadruped_spring.env.control_interface import motor_interface as mi
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented action spaces for deep reinforcement learning:
#   - "DEFAULT": classic
#   - "SYMMETRIC" legs right side and left side move symmetrically
#   - "SYMMETRIC_NO_HIP" as symmetric but hips receive action = 0

# Motor control modes:
#   - "TORQUE":
#         supply raw torques to each motor (12)
#   - "PD":
#         supply desired joint positions to each motor (12)
#         torques are computed based on the joint position/velocity error
#   - "CARTESIAN_PD":
#         supply desired foot positions for each leg (12)
#         torques are computed based on the joint position/velocity error


class MotorInterfaceCollection(CollectionBase):
    """
    Utility to collect all the implemented robot motor
    interface among action and control command.
    """

    def __init__(self):
        super().__init__()
        self.TORQUE = mi.MotorInterfaceTORQUE
        self.PD = mi.MotorInterfacePD
        self.CARTESIAN_PD = mi.MotorInterfaceCARTESIAN_PD
        self._element_type = "motor control mode"
        self._dict = {"TORQUE": self.TORQUE, "PD": self.PD, "CARTESIAN_PD": self.CARTESIAN_PD}


class ActionInterfaceCollection(CollectionBase):
    """
    Utility to collect all the implemented robot action
    interface among action in the selected action space
    and action in the default action space that has
    dimension = 12.
    """

    def __init__(self):
        self.DEFAULT = ai.DefaultActionWrapper
        self.SYMMETRIC = ai.SymmetricActionWrapper
        self.SYMMETRIC_NO_HIP = ai.SymmetricNoHipActionWrapper
        self._element_type = "action space mode"
        self._dict = {"DEFAULT": self.DEFAULT, "SYMMETRIC": self.SYMMETRIC, "SYMMETRIC_NO_HIP": self.SYMMETRIC_NO_HIP}

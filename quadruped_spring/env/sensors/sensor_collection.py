from quadruped_spring.env.sensors import robot_sensors as rs
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented observation spaces for deep reinforcement learning:
#   "DEFAULT":  IMU (base linear velocity, base orientation, base angular rate) +
#               Feet position, Feet velociites, ground reaction force
#   "ENCODER":  IMU + Joint position and velocity + ground reaction force
#   "CARTESIAN_NO_IMU": Feet position, feet velocities, ground reaction force
#   "ANGLE_NO_IMU": Joint position, joint velocities, ground reaction force
#   "CARTESIAN_ANGLE_NO_IMU": CARTESIAN_NO_IMU + ANGLE_NO_IMU


class SensorCollection(CollectionBase):
    """Utility to collect all the implemented robot sensor equipments."""

    def __init__(self):
        super().__init__()
        self._ENCODER = [rs.JointPosition, rs.JointVelocity]
        self._ENCODER_2 = [rs.LinearVelocity, rs.AngularVelocity, rs.JointPosition, rs.JointVelocity]
        self._CARTESIAN_NO_IMU = [rs.FeetPostion, rs.FeetVelocity]
        self._ARS_BASIC = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.Height,
            rs.BaseHeightVelocity,
        ]

        self._ARS_SENSOR = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.PitchRate,
            rs.Height,
            rs.BaseHeightVelocity,
        ]
        self._LANDING_SENSOR = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.PitchRate,
            rs.Height,
            rs.BaseHeightVelocity,
            rs.Landing,
        ]
        self._PPO_BASIC = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.Height,
            rs.BaseHeightVelocity,
            rs.Landing,
        ]
        self._PPO_BASIC_X = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.Height,
            rs.BaseHeightVelocity,
            rs.VelocityX,
            rs.Landing,
        ]
        self._PPO_BASIC_CONTACT = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.Height,
            rs.BaseHeightVelocity,
            rs.Landing,
            rs.BooleanContact,
        ]

        self._dict = {
            "ENCODER": self._ENCODER,
            "ENCODER_2": self._ENCODER_2,
            "CARTESIAN_NO_IMU": self._CARTESIAN_NO_IMU,
            "ARS_BASIC": self._ARS_BASIC,
            "ARS_SENSOR": self._ARS_SENSOR,
            "LANDING_SENSOR": self._LANDING_SENSOR,
            "PPO_BASIC": self._PPO_BASIC,
            "PPO_BASIC_X": self._PPO_BASIC_X,
            "PPO_BASIC_CONTACT": self._PPO_BASIC_CONTACT,
        }
        self._element_type = "sensor package"

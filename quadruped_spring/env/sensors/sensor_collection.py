from quadruped_spring.env.sensors import robot_sensors as rs
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented observation spaces for deep reinforcement learning:
#   "DEFAULT":  IMU (base linear velocity, base orientation, base angular rate) +
#               Feet position, Feet velociites, ground reaction force
#   "CARTESIAN_NO_IMU": Feet position, feet velocities, ground reaction force
#   "ANGLE_NO_IMU": Joint position, joint velocities, ground reaction force
#   "CARTESIAN_ANGLE_NO_IMU": CARTESIAN_NO_IMU + ANGLE_NO_IMU


class SensorCollection(CollectionBase):
    """Utility to collect all the implemented robot sensor equipments."""

    def __init__(self):
        super().__init__()
        self._DEFAULT = [rs.IMU, rs.FeetPostion, rs.FeetVelocity, rs.GroundReactionForce]
        self._CARTESIAN_NO_IMU = [rs.FeetPostion, rs.FeetVelocity, rs.GroundReactionForce]
        self._ANGLE_NO_IMU = [rs.JointPosition, rs.JointVelocity, rs.GroundReactionForce]
        self._CARTESIAN_ANGLE_NO_IMU = [
            rs.FeetPostion,
            rs.FeetVelocity,
            rs.JointPosition,
            rs.JointVelocity,
            rs.GroundReactionForce,
        ]
        self._dict = {
            "DEFAULT": self._DEFAULT,
            "CARTESIAN_NO_IMU": self._CARTESIAN_NO_IMU,
            "ANGLE_NO_IMU": self._ANGLE_NO_IMU,
            "CARTESIAN_ANGLE_NO_IMU": self._CARTESIAN_ANGLE_NO_IMU,
        }
        self._element_type = "sensor package"
from quadruped_spring.env.sensors import robot_sensors as rs
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented observation spaces for deep reinforcement learning:
#   "DEFAULT":  IMU (base linear velocity, base orientation, base angular rate) +
#               Feet position, Feet velociites, ground reaction force


class SensorCollection(CollectionBase):
    """Utility to collect all the implemented robot sensor equipments."""

    def __init__(self):
        super().__init__()
        self._DEFAULT = [rs.IMU, rs.FeetPostion, rs.FeetVelocity, rs.GroundReactionForce]
        self._dict = {"DEFAULT": self._DEFAULT}
        self._element_type = "sensor package"

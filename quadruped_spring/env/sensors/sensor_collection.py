from quadruped_spring.env.sensors import robot_sensors as rs
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented observation spaces for deep reinforcement learning.
# Actually only ENCODER, ENCODER_IMU and CARTESIAN are implemented.
# Follow the API for building your own sensors equipment by using the sensors already
# defined in 'robot_sensors.py' or by defining new ones.


class SensorCollection(CollectionBase):
    """Utility to collect all the implemented robot sensors equipments."""

    def __init__(self):
        super().__init__()
        self._ENCODER = [rs.JointPosition, rs.JointVelocity]
        self._ENCODER_IMU = [rs.OrientationRPY, rs.LinearVelocity, rs.AngularVelocity, rs.JointPosition, rs.JointVelocity]
        self._CARTESIAN = [rs.FeetPostion, rs.FeetVelocity]

        self._dict = {
            "ENCODER": self._ENCODER,
            "ENCODER_IMU": self._ENCODER_IMU,
            "CARTESIAN": self._CARTESIAN,
        }
        self._element_type = "sensor package"

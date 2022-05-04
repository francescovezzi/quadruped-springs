import numpy as np


class Sensor:
    """A prototype class for a generic sensor"""

    def __init__(self):
        self._data = None
        self._robot = None
        self._name = "please give me a name"

    def _check_dimension(self):
        assert np.shape(self._high) == np.shape(self._low), "high limit and low limits are different in size"
        assert np.shape(self._high) == np.shape(self._data), "Observation different from sensor observation limits"

    def _update_sensor_info(self, high, low, noise_std):
        self._high = high
        self._low = low
        self._noise_std = noise_std
        self._dim = np.shape(self._high)

    def _set_sensor(self, robot):
        """Call it at init"""
        self._robot = robot

    def _init_sensor(self, robot_config):
        self._robot_config = robot_config

    def _reset_sensor(self):
        """Call it at reset"""
        pass

    def _read_data(self):
        """Get Sensor data"""
        pass

    def _on_step(self):
        """Callback for step method"""
        pass

    def _get_data(self):
        """Get sensor data"""
        pass

import numpy as np


class Sensor:
    """A prototype class for a generic sensor"""

    def __init__(self, env):
        """Initialize Sensor."""
        self._data = None
        self._env = env
        self._robot = None
        self._name = "please give me a name"

    def _check_dimension(self):
        assert np.shape(self._high) == np.shape(self._low), "high limit and low limits are different in size"
        assert np.shape(self._high) == np.shape(self._data), "Observation different from sensor observation limits"

    def _update_sensor_info(self, high, low, noise_std):
        """Update the sensor reading."""
        self._high = high
        self._low = low
        self._noise_std = noise_std
        self._dim = np.shape(self._high)

    def _sample_noise(self):
        """Sample noise to be added to sensor reading."""
        if np.all(self._noise_std > 0.0):
            self._add_obs_noise = np.random.normal(scale=self._noise_std)
        elif np.all(self._noise_std == 0.0):
            self._add_obs_noise = np.zeros(np.shape(self._noise_std))
        else:
            raise ValueError(f"Noise standard deviation should be >= 0.0. not {self._noise_std}")

    def _set_sensor(self, robot):
        """Call it at init"""
        self._robot = robot

    def _init_sensor(self, robot_config):
        """Update config file for noise std and measurement limits."""
        self._robot_config = robot_config

    def _read_data(self):
        """Get Sensor data without noise"""
        return self._data

    def _read_dirty_data(self):
        """Get Sensor data with noise"""
        # print(self._data + self._add_obs_noise)
        if np.all(self._noise_std > 0):
            return self._data + self._add_obs_noise
        else:
            return self._data

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()

    def _get_data(self):
        """Get sensor data"""
        pass

    def _update_env(self, env):
        """Update the environment reference"""
        self._env = env


class SensorList:
    """Manage all the robot sensors"""

    def __init__(self, sensor_list, env):
        if not isinstance(sensor_list, list):
            raise ValueError("Please use a list of sensors. Also if it is just one.")
        self._sensor_list = sensor_list
        self._env = env

    def _compute_obs_dim(self):
        dim = 0
        for s in self._sensor_list:
            dim += np.sum(np.array(s._dim))
        return dim

    def get_obs_dim(self):
        return self._obs_dim

    def _get_high_limits(self):
        high = []
        for s in self._sensor_list:
            high.append(s._high.flatten())
        return np.concatenate(high)

    def _get_low_limits(self):
        low = []
        for s in self._sensor_list:
            low.append(np.array(s._low).flatten())
        return np.concatenate(low)

    def get_obs(self):
        obs = {}
        for s in self._sensor_list:
            obs[s._name] = s._read_data()
        return obs

    def get_noisy_obs(self):
        obs = {}
        for s in self._sensor_list:
            obs[s._name] = s._read_dirty_data()
        return obs

    def _on_step(self):
        for s in self._sensor_list:
            s._on_step()

    def _reset(self, robot):
        for s in self._sensor_list:
            s._set_sensor(robot)
            s._reset_sensor()
        self._obs_dim = self._compute_obs_dim()

    def _init(self, robot_config):
        for idx, s in enumerate(self._sensor_list):
            self._sensor_list[idx] = s(self._env)
        for s in self._sensor_list:
            s._init_sensor(robot_config)
            s._update_sensor_info()

    def _turn_on(self, robot):
        for s in self._sensor_list:
            s._set_sensor(robot)

    def _reinit(self, env):
        for s in self._sensor_list:
            s._update_env(env)

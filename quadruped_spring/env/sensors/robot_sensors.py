import numpy as np

from quadruped_spring.env.sensors.sensor import Sensor


class BooleanContact(Sensor):
    """Boolean variables specifying if the feet are in contact with ground"""

    def __init__(self):
        super().__init__()
        self._name = "BoolContatc"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.CONTACT_BOOL_HIGH,
            low=self._robot_config.CONTACT_BOOL_LOW,
            noise_std=self._robot_config.CONTACT_BOOL_NOISE,
        )

    def _get_data(self):
        _, _, _, feet_in_contact = self._robot.GetContactInfo()
        self._data = np.array(feet_in_contact)

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class GroundReactionForce(Sensor):
    """Boolean variables specifying if the feet are in contact with ground"""

    def __init__(self):
        super().__init__()
        self._name = "Groundreaction"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.CONTACT_FORCE_HIGH,
            low=self._robot_config.CONTACT_FORCE_LOW,
            noise_std=self._robot_config.CONTACT_FORCE_NOISE,
        )

    def _get_data(self):
        _, _, normal_force, _ = self._robot.GetContactInfo()
        self._data = np.array(normal_force)

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class JointPosition(Sensor):
    """Joint Configuration"""

    def __init__(self):
        super().__init__()
        self._name = "Encoder"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.JOINT_ANGLES_HIGH,
            low=self._robot_config.JOINT_ANGLES_LOW,
            noise_std=self._robot_config.JOINT_ANGLES_NOISE,
        )

    def _get_data(self):
        angles = self._robot.GetMotorAngles()
        self._data = angles

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class JointVelocity(Sensor):
    """Joint Vecloity"""

    def __init__(self):
        super().__init__()
        self._name = "JointVelocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.JOINT_VELOCITIES_HIGH,
            low=self._robot_config.JOINT_VELOCITIES_LOW,
            noise_std=self._robot_config.JOINT_VELOCITIES_NOISE,
        )

    def _get_data(self):
        velocities = self._robot.GetMotorVelocities()
        self._data = velocities

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class FeetPostion(Sensor):
    """Feet position in leg frame"""

    def __init__(self):
        super().__init__()
        self._name = "FeetPosition"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.FEET_POS_HIGH,
            low=self._robot_config.FEET_POS_LOW,
            noise_std=self._robot_config.FEET_POS_NOISE,
        )

    def _get_data(self):
        feet_pos, _ = self._robot.ComputeFeetPosAndVel()
        self._data = feet_pos

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class FeetVelocity(Sensor):
    """Feet velocity in leg frame"""

    def __init__(self):
        super().__init__()
        self._name = "FeetVelocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.FEET_VEL_HIGH,
            low=self._robot_config.FEET_VEL_LOW,
            noise_std=self._robot_config.FEET_VEL_NOISE,
        )

    def _get_data(self):
        _, feet_vel = self._robot.ComputeFeetPosAndVel()
        self._data = feet_vel

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class IMU(Sensor):
    """base linear velocity + base_orientation (rpy) + base orientation rate (rpy)"""

    def __init__(self):
        super().__init__()
        self._name = "IMU"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.IMU_HIGH,
            low=self._robot_config.IMU_LOW,
            noise_std=self._robot_config.IMU_NOISE,
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()
        ang_vel = self._robot.GetTrueBaseRollPitchYawRate()
        base_orientation = self._robot.GetBaseOrientationRollPitchYaw()
        self._data = np.concatenate((lin_vel, base_orientation, ang_vel))

    def _read_data(self):
        return self._data

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()


class SensorList():
    """Manage all the robot sensors"""

    def __init__(self, sensor_list):
        if not isinstance(sensor_list, list):
            raise ValueError("Please use a list of sensors. Also if it is just one")
        self._sensor_list = sensor_list

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

    def _on_step(self):
        for s in self._sensor_list:
            s._get_data()

    def _reset(self, robot):
        for s in self._sensor_list:
            s._set_sensor(robot)
            s._reset_sensor()
        self._obs_dim = self._compute_obs_dim()
        
    def _init(self, robot_config):
        for idx, s in enumerate(self._sensor_list):
            self._sensor_list[idx] = s()
        for s in self._sensor_list:
            s._init_sensor(robot_config)
            s._update_sensor_info()
            
    def _turn_on(self, robot):
        for s in self._sensor_list:
            s._set_sensor(robot)

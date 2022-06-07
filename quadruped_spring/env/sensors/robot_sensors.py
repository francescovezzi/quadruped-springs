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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class LinearVelocity(Sensor):
    """Base linear velocity."""

    def __init__(self):
        super().__init__()
        self._name = "Base Linear Velocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH,
            low=self._robot_config.VEL_LIN_LOW,
            noise_std=self._robot_config.VEL_LIN_NOISE,
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()
        self._data = lin_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class AngularVelocity(Sensor):
    """Base angular velocity."""

    def __init__(self):
        super().__init__()
        self._name = "Base Angular Velocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_ANG_HIGH,
            low=self._robot_config.VEL_ANG_LOW,
            noise_std=self._robot_config.VEL_ANG_NOISE,
        )

    def _get_data(self):
        ang_vel = self._robot.GetBaseAngularVelocity()
        self._data = ang_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


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

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class Height(Sensor):
    """robot height."""

    def __init__(self):
        super().__init__()
        self._name = "Height"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.HEIGHT_HIGH,
            low=self._robot_config.HEIGHT_LOW,
            noise_std=self._robot_config.HEIGHT_NOISE,
        )

    def _get_data(self):
        height = self._robot.GetBasePosition()[2]
        self._data = height

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class DesiredBaseLinearVelocityXZ(Sensor):
    """robot height."""

    def __init__(self):
        super().__init__()
        self._name = "Desired base linear velocity xz plane"
        self._desired_velocity = np.array([0.0, 0.01])

    def get_desired_velocity(self):
        return self._desired_velocity

    def set_desired_velocity(self, vel):
        self._desired_velocity = vel

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=np.array([0.0, 0.0]),
            low=np.array([0.0, 0.0]),
            noise_std=np.array([0.0, 0.0]),
        )

    def _get_data(self):
        self._data = self._desired_velocity

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class Quaternion(Sensor):
    """base_orientation (quaternion)"""

    def __init__(self):
        super().__init__()
        self._name = "Quaternion"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.QUATERNION_HIGH,
            low=self._robot_config.QUATERNION_LOW,
            noise_std=self._robot_config.QUATERNION_NOISE,
        )

    def _get_data(self):
        base_orientation = self._robot.GetBaseOrientation()
        self._data = base_orientation

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class Pitch(Sensor):
    """pitch angle."""

    def __init__(self):
        super().__init__()
        self._name = "Pitch"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.PITCH_HIGH,
            low=self._robot_config.PITCH_LOW,
            noise_std=self._robot_config.PITCH_NOISE,
        )

    def _get_data(self):
        pitch_orientation = self._robot.GetBaseOrientationRollPitchYaw()[1]
        self._data = pitch_orientation

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class PitchRate(Sensor):
    """pitch orientation rate."""

    def __init__(self):
        super().__init__()
        self._name = "Pitch rate"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.PITCH_RATE_HIGH,
            low=self._robot_config.PITCH_RATE_LOW,
            noise_std=self._robot_config.PITCH_RATE_NOISE,
        )

    def _get_data(self):
        pitch_orientation_rate = self._robot.GetTrueBaseRollPitchYawRate()[1]
        self._data = pitch_orientation_rate

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class LinearVelocity2D(Sensor):
    """Base linear velocity."""

    def __init__(self):
        super().__init__()
        self._name = "Base Linear Velocity xz plane"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH[[0, 2]],
            low=self._robot_config.VEL_LIN_LOW[[0, 2]],
            noise_std=self._robot_config.VEL_LIN_NOISE[[0, 2]],
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()[[0, 2]]
        self._data = lin_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class SensorList:
    """Manage all the robot sensors"""

    def __init__(self, sensor_list):
        if not isinstance(sensor_list, list):
            raise ValueError("Please use a list of sensors. Also if it is just one.")
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

    def get_noisy_obs(self):
        obs = {}
        for s in self._sensor_list:
            obs[s._name] = s._read_dirty_data()
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

    def get_desired_velocity(self):
        for s in self._sensor_list:
            if isinstance(s, DesiredBaseLinearVelocityXZ):
                return s.get_desired_velocity()
        raise ValueError("Desired Velocity not specified.")

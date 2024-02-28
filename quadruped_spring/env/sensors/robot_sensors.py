import numpy as np
from scipy.spatial.transform import Rotation as R

from quadruped_spring.env.sensors.sensor import Sensor
from quadruped_spring.env.tasks.task_base import TaskJumpingDemo


class BooleanContact(Sensor):
    """Boolean variables specifying if the feet are in contact with ground"""

    def __init__(self, env):
        super().__init__(env)
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


class Height(Sensor):
    """robot height."""

    def __init__(self, env):
        super().__init__(env)
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


class JointPosition(Sensor):
    """Joint Configuration"""

    def __init__(self, env):
        super().__init__(env)
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


class JointVelocity(Sensor):
    """Joint Vecloity"""

    def __init__(self, env):
        super().__init__(env)
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


class FeetPostion(Sensor):
    """Feet position in leg frame"""

    def __init__(self, env):
        super().__init__(env)
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


class FeetVelocity(Sensor):
    """Feet velocity in leg frame"""

    def __init__(self, env):
        super().__init__(env)
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


class LinearVelocity(Sensor):
    """Base linear velocity."""

    def __init__(self, env):
        super().__init__(env)
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


class Landing(Sensor):
    """Robot landing sensor detection."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "is landing"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=np.array([1]),
            low=np.array([0]),
            noise_std=np.array([0]),
        )

    def _get_data(self):
        # if self._env.task_env == "JUMPING_IN_PLACE_DEMO":
        #     self._data = self._env.task.demo_is_landing
        # else:
        self._data = np.array([self._env.task._switched_controller])

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()

class Jumping(Sensor):
    """Robot landing sensor detection."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "is jumping"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=np.array([1]),
            low=np.array([0]),
            noise_std=np.array([0]),
        )

    def _get_data(self):
        self._data = np.array([self._env.task.is_jumping])

    def _reset_sensor(self):
        self._get_data()

    def _on_step(self):
        self._get_data()

class VelocityX(Sensor):
    """Base Velocity X."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Base Height Velocity X"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH[0],
            low=self._robot_config.VEL_LIN_LOW[0],
            noise_std=self._robot_config.VEL_LIN_NOISE[0],
        )

    def _get_data(self):
        height_vel = self._robot.GetBaseLinearVelocity()[0]
        self._data = height_vel


class AngularVelocity(Sensor):
    """Base angular velocity."""

    def __init__(self, env):
        super().__init__(env)
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


class Quaternion(Sensor):
    """base_orientation (quaternion)"""

    def __init__(self, env):
        super().__init__(env)
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


class Pitch(Sensor):
    """pitch angle."""

    def __init__(self, env):
        super().__init__(env)
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


class PitchRate(Sensor):
    """pitch orientation rate."""

    def __init__(self, env):
        super().__init__(env)
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


class OrientationRPY(Sensor):
    """orientation Roll Pitch Yaw."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Orientation Roll Pitch Yaw"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.ORIENT_RPY_HIGH,
            low=self._robot_config.ORIENT_RPY_LOW,
            noise_std=self._robot_config.ORIENT_RPY_NOISE,
        )

    def _get_data(self):
        orientation = self._robot.GetBaseOrientationRollPitchYaw()
        self._data = orientation


class BaseHeightVelocity(Sensor):
    """Base height velocity."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Base Linear Velocity z direction"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH[2],
            low=self._robot_config.VEL_LIN_LOW[2],
            noise_std=self._robot_config.VEL_LIN_NOISE[2],
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()[2]
        self._data = lin_vel


class PitchBackFlip(Sensor):
    def __init__(self, env):
        super().__init__(env)
        self._name = "Pitch-BackFlip"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.PITCH_HIGH, low=self._robot_config.PITCH_LOW, noise_std=self._robot_config.PITCH_NOISE
        )

    @staticmethod
    def _get_pitch(env):
        rot = R.from_quat(env.robot.GetBaseOrientation())
        euler = rot.as_euler("yxz", degrees=False)
        pitch = -euler[0]
        if pitch < 0 and env.task._switched_controller:
            pitch = 2 * np.pi + pitch
        return pitch

    def get_pitch(self):
        rot = R.from_quat(self._robot.GetBaseOrientation())
        euler = rot.as_euler("yxz", degrees=False)
        pitch = -euler[0]
        if pitch < 0 and self._env.task._switched_controller:
            pitch = 2 * np.pi + pitch
        return pitch

    def _get_data(self):
        self._data = self.get_pitch()

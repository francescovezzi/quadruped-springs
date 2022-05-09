import numpy as np
from scipy.spatial.transform import Rotation as R


class TaskBase:
    """Prototype class for a generic task"""

    def __init__(self):
        pass

    def _reset(self, env):
        """reset task and initialize task variables"""
        self._env = env

    def _on_step(self):
        """update task variables"""
        pass

    def _reward(self):
        """Return the reward funciton"""
        pass

    def _reward_end_episode(self, reward):
        """add bonus and malus to the actual reward at the end of the episode"""
        pass

    def _terminated(self):
        """return boolean specifying whether episode is terminated"""
        pass


class TaskJumping(TaskBase):
    """Generic Jumping Task"""

    def __init__(self):
        super().__init__()

    def _reset(self, env):
        super()._reset(env)
        robot = self._env.robot
        self._all_feet_in_the_air = False
        self._time_take_off = self._env.get_sim_time()
        self._robot_pose_take_off = robot.GetBasePosition()
        self._init_height = self._robot_pose_take_off[2]
        self._robot_orientation_take_off = robot.GetBaseOrientationRollPitchYaw()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_yaw = 0.0
        self._max_roll = 0.0
        self._relative_max_height = 0.0
        self._max_vel_err = 0.0

    def _on_step(self):
        pos_abs = np.array(self._env.robot.GetBasePosition())
        vel_abs = self._env.robot.GetBaseLinearVelocity()
        orient_rpy = np.array(self._env.robot.GetBaseOrientationRollPitchYaw())
        if self._env.robot._is_flying():
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self._env.get_sim_time()
                self._robot_pose_take_off = pos_abs
                self._robot_orientation_take_off = orient_rpy
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self._env.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)
            self._all_feet_in_the_air = False

        delta_height = max(pos_abs[2] - self._init_height, 0.0)
        self._relative_max_height = max(self._relative_max_height, delta_height)
        roll, _, yaw = orient_rpy
        self._max_yaw = max(np.abs(yaw), self._max_yaw)
        self._max_roll = max(np.abs(roll), self._max_roll)
        vel_module = np.dot(vel_abs, vel_abs)
        if vel_module > 0.2:
            vel_err = np.dot(vel_abs / vel_module, np.array([0,0,1]))
            self._max_vel_err = max(vel_err, self._max_vel_err)

    def is_fallen(self, dot_prod_min=0.85):
        """Decide whether the quadruped has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.10 meter -> see config file), the quadruped
        is considered fallen.

        Returns:
          Boolean value that indicates whether the quadruped has fallen.
        """
        base_rpy = self._env.robot.GetBaseOrientationRollPitchYaw()
        orientation = self._env.robot.GetBaseOrientation()
        rot_mat = self._env._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self._env.robot.GetBasePosition()
        return (
            np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < dot_prod_min
            or pos[2] < self._env._robot_config.IS_FALLEN_HEIGHT
        )

    def _not_allowed_contact(self):
        """
        Return True if the robot is performing some not allowed contact
        as touching the ground with knees
        """
        _, num_invalid_contacts, _, _ = self._env.robot.GetContactInfo()

        return num_invalid_contacts

    def _terminated(self):
        return self.is_fallen() or self._not_allowed_contact()

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
        # self._init_height = robot._robot_config.INIT_HEIGHT
        self._init_height = self._robot_pose_take_off[2]
        self._robot_orientation_take_off = robot.GetBaseOrientationRollPitchYaw()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_yaw = 0.0
        self._max_pitch = 0.0
        self._max_roll = 0.0
        self._relative_max_height = 0.0
        self._max_delta_x = 0.0
        self._max_delta_y = 0.0
        self._max_vel_err = 1.0
        self._base_acc = np.zeros(3)
        self._new_action = self._old_action = self._env.get_last_action()
        self._max_delta_action = 0.0

    def _on_step(self):
        self._update_actions()
        self._update_pose()
        self._compute_pose_info()
        self._compute_jumping_info()

    def _update_pose(self):
        self._pos_abs = np.array(self._env.robot.GetBasePosition())
        self._vel_abs = self._env.robot.GetBaseLinearVelocity()
        self._orient_rpy = np.array(self._env.robot.GetBaseOrientationRollPitchYaw())

    def _compute_pose_info(self):
        self._compute_position_info()
        self._compute_orientation_info()
        self._compute_velocity_info()

    def _compute_position_info(self):
        x, y, z = self._pos_abs
        delta_height = max(z - self._init_height, 0.0)
        self._relative_max_height = max(self._relative_max_height, delta_height)
        self._max_delta_x = max(abs(x), self._max_delta_x)
        self._max_delta_y = max(abs(y), self._max_delta_y)

    def _compute_orientation_info(self):
        roll, pitch, yaw = self._orient_rpy
        self._max_yaw = max(np.abs(yaw), self._max_yaw)
        self._max_roll = max(np.abs(roll), self._max_roll)
        self._max_pitch = max(np.abs(pitch), self._max_pitch)

    def _compute_velocity_info(self):
        vel_abs = self._vel_abs
        vel_module = np.sqrt(np.dot(vel_abs, vel_abs))
        if vel_module > 0.01:
            vel_err = 1 - np.dot(vel_abs / vel_module, np.array([0, 0, 1]))
            self._max_vel_err = max(vel_err, self._max_vel_err)

    def _compute_jumping_info(self):
        pos_abs = self._pos_abs
        orient_rpy = self._orient_rpy
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

    def is_fallen(self, dot_prod_min=0.85):
        """Decide whether the quadruped has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.10 meter -> see config file), the quadruped
        is considered fallen.

        Returns:
          Boolean value that indicates whether the quadruped has fallen.
        """
        orientation = self._env.robot.GetBaseOrientation()
        rot_mat = self._env._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self._pos_abs
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

    def print_info(self):
        print(f"max forward distance -> {self._max_forward_distance:.3f}")
        print(f"max peak height -> {(self._init_height + self._relative_max_height):.3f}")

    def _update_actions(self):
        self._old_action = self._new_action
        self._new_action = self._env.get_last_action()
        if not self._env.robot._is_flying():
            self._delta_action = np.abs(self._new_action - self._old_action)
            self._max_delta_action = max(self._max_delta_action, np.amax(self._delta_action))

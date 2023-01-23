import numpy as np
from scipy.spatial.transform import Rotation as R

from quadruped_spring.env.wrappers.get_demonstration_wrapper import GetDemonstrationWrapper as DemoWrapper


class TaskBase:
    """Prototype class for a generic task"""

    def __init__(self, env):
        self._env = env

    def _reset(self):
        """reset task and initialize task variables"""
        pass

    def _on_step(self):
        """update task variables"""
        pass

    def _reward(self):
        """Return the reward funciton"""
        return 0

    def _reward_end_episode(self):
        """add bonus and malus to the actual reward at the end of the episode"""
        return 0

    def _terminated(self):
        """return boolean specifying whether episode is terminated"""
        pass


class TaskJumping(TaskBase):
    """Generic Jumping Task"""

    def __init__(self, env):
        super().__init__(env)

    def _reset(self):
        self._reset_params()

    def _reset_params(self):
        self._switched_controller = False
        self._all_feet_in_the_air = False
        self._time_take_off = self._env.get_sim_time()
        self._robot_pose_take_off = self._env.robot.GetBasePosition()
        self._init_height = self._robot_pose_take_off[2]
        self._robot_orientation_take_off = self._env.robot.GetBaseOrientationRollPitchYaw()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_pitch = 0.0
        self._relative_max_height = 0.0
        self._max_delta_x = 0.0
        self._old_torque = self._new_torque = self._env.robot.GetMotorTorques()
        self._max_height = 0.0
        self._x_actual = self._x_previos = 0.0
        self.new_config = self._env.robot.GetMotorAngles()

    def _on_step(self):
        self._task_jump_take_off()
        self._update_torques()
        self._update_pose()
        self._compute_pose_info()
        self._compute_jumping_info()

    def _update_torques(self):
        self._old_torque = self._new_torque
        self._new_torque = self._env.robot.GetMotorTorques()

    def _update_pose(self):
        self._pos_abs = np.array(self._env.robot.GetBasePosition())
        self._vel_abs = self._env.robot.GetBaseLinearVelocity()
        self._orient_rpy = np.array(self._env.robot.GetBaseOrientationRollPitchYaw())

    def _compute_pose_info(self):
        self._compute_position_info()
        self._compute_orientation_info()

    def _compute_position_info(self):
        x, _, z = self._pos_abs
        delta_height = max(z - self._init_height, 0.0)
        self._relative_max_height = max(self._relative_max_height, delta_height)
        self._max_height = max(abs(z), self._max_height)
        self._max_delta_x = max(abs(x), self._max_delta_x)

    def _compute_orientation_info(self):
        _, pitch, _ = self._orient_rpy
        self._max_pitch = max(np.abs(pitch), self._max_pitch)

    def _compute_jumping_info(self):
        if self._env.robot._is_flying():
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self._env.get_sim_time()
                self._robot_pose_take_off = self._pos_abs
                self._robot_orientation_take_off = self._orient_rpy
            else:
                self.compute_max_forward_distance()
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self._env.get_sim_time() - self._time_take_off, self._max_flight_time)
                self.compute_max_forward_distance()
                self._all_feet_in_the_air = False
            else:
                self._max_forward_distance = 0

    def compute_max_forward_distance(self):
        """Compute forward distance according to local frame (starting at take off)"""
        rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
        translation = -self._robot_pose_take_off
        pos_relative = self._pos_abs + translation
        pos_relative = pos_relative @ rotation_matrix
        self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

    def _is_fallen_ground(self):
        return self._pos_abs[2] < self._env._robot_config.IS_FALLEN_HEIGHT

    def _is_fallen_orientation(self):
        orientation = self._env.robot.GetBaseOrientation()
        rot_mat = self._env._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def is_fallen(self):
        """Decide whether the quadruped has fallen."""

        return self._is_fallen_orientation() and self._is_fallen_ground()

    def _not_allowed_contact(self):
        """
        Return True if the robot is performing some not allowed contact
        as touching the ground with knees
        """
        _, num_invalid_contacts, _, _ = self._env.robot.GetContactInfo()

        return num_invalid_contacts

    def _terminated(self):
        return self.is_fallen() or self._not_allowed_contact()

    def _get_torque_diff(self):
        return self._old_torque - self._new_torque

    def _task_jump_take_off(self):
        """Switch controller if the robot is starting the task jump."""
        if not self._switched_controller and self._env.robot._is_flying() and self.compute_time_for_peak_heihgt() > 0.06:
            self._switched_controller = True

    def compute_time_for_peak_heihgt(self):
        """Compute the time the robot needs to reach the maximum height"""
        _, _, vz = self._env.robot.GetBaseLinearVelocity()
        return vz / 9.81

    def is_switched_controller(self):
        return self._switched_controller

    def turn_off_landing_mode(self):
        self._switched_controller = False


class TaskJumpingDemo(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        # self.demo_path = os.path.join(os.path.dirname(qs.__file__), "demonstrations", "demo_list_imitate.npy")
        self.demo_list = np.load(self.demo_path)
        self.demo_length = np.shape(self.demo_list)[0]
        self.action_dim = self._env.get_action_dim()
        self.max_err = np.sqrt(2**2 * self.action_dim)
        self.demo_counter = 0

    def _reset(self):
        if self._env.robot_desired_state is None:
            self.demo_counter = 0
        self.demo_is_landing = self.get_demo()[-1]
        self.delta_demo = self.demo_length - self.demo_counter
        return super()._reset()

    def get_demo(self):
        demo = self.demo_list[self.demo_counter, :]
        return DemoWrapper.read_demo(
            demo, action_dim=self._env.get_action_dim(), num_joints=self._env.get_robot_config().NUM_MOTORS
        )

    @staticmethod
    def norm(vec):
        return np.dot(vec, vec)

    def _reward(self):
        (
            demo_action,
            demo_joint_pos,
            demo_joint_vel,
            demo_base_pos,
            demo_base_or,
            demo_base_lin_vel,
            demo_base_ang_vel,
            demo_is_landing,
        ) = self.get_demo()
        self.demo_is_landing = demo_is_landing
        self.demo_counter += 1
        actual_action = self._env.get_last_action()
        norm = np.sqrt(self.norm(demo_action - actual_action))
        rew = np.exp(-0.35 * norm)
        return rew / self.delta_demo

    def _terminated(self):
        return super()._terminated() or self.demo_counter == self.demo_length

    def _reward_end_episode(self):
        reward = 0
        return reward

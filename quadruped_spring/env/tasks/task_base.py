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
        self._on_step()

    def _reset_params(self):
        self._switched_controller = False
        self._all_feet_in_the_air = False
        self._time_take_off = self._env.get_sim_time()
        self._robot_pose_take_off = np.array(self._env.robot.GetBasePosition())
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

    def compute_jumping_distance(self):
        """Compute forward distance according to local frame (starting at take off)"""
        rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
        translation = -self._robot_pose_take_off
        pos_relative = self._pos_abs + translation
        pos_relative = pos_relative @ rotation_matrix
        jump_distance = max(pos_relative[0], 0)
        return jump_distance

    def compute_max_forward_distance(self):
        """Update max forward distance"""
        jump_distance = self.compute_jumping_distance()
        self._max_forward_distance = max(jump_distance, self._max_forward_distance)

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
        self.action_dim = self._env.action_dim
        self.max_err = np.sqrt(2**2 * self.action_dim)
        self.demo_counter = 0

    def _reset(self):
        if self._env.robot_desired_state is None:
            self.demo_counter = 0
        self.demo_is_landing = self.get_demo()[-1]
        self.delta_demo = self.demo_length - self.demo_counter
        return super()._reset()

    def get_demo(self):
        # print(f'demo counter: {self.demo_counter}')
        demo = self.demo_list[self.demo_counter, :]
        return DemoWrapper.read_demo(demo, action_dim=self.action_dim, num_joints=self._env.get_robot_config().NUM_MOTORS)

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
        return 0

    def set_demo_counter(self, value):
        self.demo_counter = value

class TaskContinuousJumping(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self.jump_limit = 0.5
        self.time_limit = 1.0

    def _reset_params(self):
        super()._reset_params()
        self.cumulative_fwd = 0.0
        self.cumulative_flight_time = 0.0
        self.jump_counter = 0
        self.is_jumping = False
        # self.first_jump = True

    def detect_jumping(self):
        """Detect whether the robot is jumping."""
        if self._env.robot._is_flying() and self.compute_time_for_peak_heihgt() > 0.06:
            return True
        else:
            return False

    def _compute_jumping_info(self):
        if self._env.robot._is_flying():
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self._env.get_sim_time()
                self._robot_pose_take_off = self._pos_abs
                self._robot_orientation_take_off = self._orient_rpy
                self.is_jumping = self.detect_jumping()
            # else:
            # self.compute_max_forward_distance()
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self._env.get_sim_time() - self._time_take_off, self._max_flight_time)
                self.compute_max_forward_distance()
                self.update_end_jump()
                self._all_feet_in_the_air = False
                self.is_jumping = False
            # else:
            #     self.update_end_jump()

    def update_end_jump(self):
        self.cumulative_fwd += min(self._max_forward_distance, self.jump_limit)
        self.cumulative_flight_time += min(self._max_flight_time, self.time_limit)
        # n_jump = max(self.jump_counter, 1)
        # fwd = min(self._max_forward_distance, self.jump_limit)
        # if self.first_jump and fwd < 0.05:  # ignore first jump
        #     self.first_jump = False
        #     return

        # ft = min(self._max_flight_time, self.time_limit)
        # self.cumulative_fwd += (fwd - self.cumulative_fwd) / n_jump
        # self.cumulative_flight_time += (ft - self.cumulative_flight_time) / n_jump
        # self._max_forward_distance = 0.0
        # self._max_flight_time = 0.0
        # self.jump_counter += 1

    def get_jumping(self):
        return self.is_jumping


class TaskContinuousJumping2(TaskJumping):
    def __init__(self, env):
        super().__init__(env)
        self.jump_limit = 0.5
        self.height_limit = 0.5
        self.height_weight = 0.3
        self.fwd_weight = 0.7
        self.performance_bound = 0.85
        self.safe_log2 = lambda m: np.log2(m, out=np.zeros_like(m), where=(m != 0))

    def _reset_params(self):
        super()._reset_params()
        self.fwd_array = np.empty(0)
        self.height_array = np.empty(0)
        self.performance_array = np.empty(0)
        self.jump_counter = 0
        self.good_jump_counter = 0
        self.max_jump_height = 0
        self.max_jump_fwd = 0
        self.is_jumping = False
        self.first_jump = True
        self.end_jump = False

    def restart_jump_performance_variables(self):
        self.max_jump_height = 0
        self.max_jump_fwd = 0

    def detect_jumping(self):
        """Detect whether the robot is jumping."""
        if self._env.robot._is_flying() and self.compute_time_for_peak_heihgt() > 0.06:
            return True
        else:
            return False
        
    def is_rested(self):
        vel = self._env.robot.GetBaseLinearVelocity()
        vel_module = np.sqrt(np.dot(vel, vel))
        return vel_module < 0.2

    def _compute_jumping_info(self):
        self.end_jump = False
        if self._env.robot._is_flying():
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self._env.get_sim_time()
                self._robot_pose_take_off = self._pos_abs
                self.max_jump_height = self._pos_abs[2]
                self._robot_orientation_take_off = self._orient_rpy
                self.is_jumping = self.detect_jumping()
                self.restart_jump_performance_variables()
            else:
                self.max_jump_height = max(self.max_jump_height, self._pos_abs[2])
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self._env.get_sim_time() - self._time_take_off, self._max_flight_time)
                self.update_end_jump()
                self._all_feet_in_the_air = False
                self.is_jumping = False

    def update_end_jump(self):
        if not self.first_jump:  # Ignore the first jump.
            self.jump_counter += 1
            fwd = np.asarray([min(self.compute_jumping_distance(), self.jump_limit)])
            height = np.asarray([min(self.max_jump_height, self.height_limit)])
            self.fwd_array = np.concatenate((self.fwd_array, fwd))
            self.height_array = np.concatenate((self.height_array, height))
            perf = self.fwd_weight * fwd / self.jump_limit + self.height_weight * height / self.height_limit
            if perf >= self.performance_bound:
                self.good_jump_counter += 1
            self.performance_array = np.concatenate((self.performance_array, perf))
            self.end_jump = True
        else:
            self.first_jump = False

    def get_jumping(self):
        return self.is_jumping

    def get_cumulative_fwd(self):
        return np.sum(self.fwd_array)

    def get_cumulative_height(self):
        return np.sum(self.height_array)

    def get_average_fwd(self):
        if self.jump_counter == 0:
            return 0
        return np.sum(self.fwd_array) / self.jump_counter

    def get_average_height(self):
        if self.jump_counter == 0:
            return 0
        return np.sum(self.height_array) / self.jump_counter

    def get_entropy_fwd(self):
        if self.jump_counter == 0 or np.sum(self.fwd_array) < 0.05:
            return 0
        fwd = np.copy(self.fwd_array)
        while np.size(fwd) < 3:
            fwd = np.concatenate((fwd, np.asarray([0])))
        p = fwd / np.sum(fwd)
        return -np.sum(p * self.safe_log2(p)) / np.log2(np.size(p))

    def get_performance_array(self):
        return self.performance_array

    def get_actual_fwd(self):
        if self.is_jumping:
            return self.compute_jumping_distance()
        else:
            return 0

    def get_avg_performance(self):
        performance_array = np.copy(self.performance_array)

        while np.size(performance_array) < 3:
            performance_array = np.concatenate((performance_array, np.asarray([0])))

        return np.average(performance_array)

class TaskJumpingDemo2(TaskContinuousJumping2):
    def __init__(self, env):
        super().__init__(env)
        # self.demo_path = os.path.join(os.path.dirname(qs.__file__), "demonstrations", "demo_list_imitate.npy")
        self.demo_list = np.load(self.demo_path)
        self.demo_length = np.shape(self.demo_list)[0]
        self.action_dim = self._env.action_dim
        self.max_err = np.sqrt(2**2 * self.action_dim)
        self.demo_counter = 0

    def _reset(self):
        if self._env.robot_desired_state is None:
            self.demo_counter = 0
        self.demo_is_landing = self.get_demo()[-1]
        self.delta_demo = self.demo_length - self.demo_counter
        return super()._reset()

    def get_demo(self):
        # print(f'demo counter: {self.demo_counter}')
        demo = self.demo_list[self.demo_counter, :]
        return DemoWrapper.read_demo(demo, action_dim=self.action_dim, num_joints=self._env.get_robot_config().NUM_MOTORS)

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
        return 0

    def set_demo_counter(self, value):
        self.demo_counter = value
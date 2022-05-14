import numpy as np


class Springs:
    """Define the joint level springs in parallel with motors."""

    def __init__(self, robot_config):
        self._robot_config = robot_config
        self._init_springs()

    def _init_springs(self):
        self._stiffness_nominal = self._robot_config.SPRINGS_STIFFNESS
        self._damping_nominal = self._robot_config.SPRINGS_DAMPING
        self._rest_angles = self._robot_config.SPRINGS_REST_ANGLE

    def set_stiffness(self, new_stiffness):
        self._stiffness_nominal = new_stiffness

    def set_damping(self, new_damping):
        self._damping_nominal = new_damping

    def set_rest_angles(self, new_angles):
        self._rest_angles = new_angles

    def get_spring_nominal_params(self):
        return self._stiffness_nominal, self._damping_nominal, self._rest_angles

    def get_spring_real_params(self, motor_angles):
        """Get the spring params: stiffness, damping, rest_angles."""
        k, b = self.get_real_stiffness_and_damping(motor_angles)
        rest_angles = np.array(self._rest_angles * self._robot_config.NUM_LEGS)
        return k, b, rest_angles

    def _get_real_stiffness_one_leg(self, leg_angles, robot_side):
        """
        Return the real stiffness for the springs for the leg on the right or left side.
        Note that the implemented springs works only in compression.
        """
        hip_angle, thigh_angle, calf_angle = leg_angles
        k_hip, k_thigh, k_calf = self._stiffness_nominal.copy()
        b_hip, b_thigh, b_calf = self._damping_nominal.copy()
        hip_rest, thigh_rest, calf_rest = self._rest_angles
        if robot_side == "left":
            hip_cond = hip_angle < hip_rest
        elif robot_side == "right":
            hip_cond = hip_angle > hip_rest
        else:
            raise ValueError(f'robot sides should "right" or "left". not {robot_side}')
        if hip_cond:
            k_hip = 0
            b_hip = 0
        if thigh_angle < thigh_rest:
            k_thigh = 0
            b_thigh = 0
        if calf_angle > calf_rest:
            k_calf = 0
            b_calf = 0

        k_real = [k_hip, k_thigh, k_calf]
        b_real = [b_hip, b_thigh, b_calf]
        return k_real, b_real

    def get_real_stiffness_and_damping(self, motor_angles):
        """Get the real stiffness for all the joint level springs."""
        side_map = ["right", "left"] * 2
        real_stiffness = np.zeros(self._robot_config.NUM_MOTORS)
        real_damping = np.zeros(self._robot_config.NUM_MOTORS)
        for i in range(self._robot_config.NUM_LEGS):
            leg_motor_angles = motor_angles[3 * i : 3 * (i + 1)]
            leg_side = side_map[i]
            k_real_leg, b_real_leg = self._get_real_stiffness_one_leg(leg_motor_angles, leg_side)
            real_stiffness[3 * i : 3 * (i + 1)] = k_real_leg
            real_damping[3 * i : 3 * (i + 1)] = b_real_leg
        return real_stiffness, real_damping

    def compute_spring_torques(self, motor_angles, motor_velocities):
        k, b = self.get_real_stiffness_and_damping(motor_angles)
        spring_torques = -k * (motor_angles - self._rest_angles) - b * motor_velocities
        return spring_torques

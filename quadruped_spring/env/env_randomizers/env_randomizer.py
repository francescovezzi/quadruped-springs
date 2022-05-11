from email.mime import base

import numpy as np

from quadruped_spring.env.env_randomizers.env_randomizer_base import EnvRandomizerBase
from quadruped_spring.utils.timer import Timer

# Relative range.
BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
MAX_SETTLING_ACTION_DISTURBANCE = (0.05, 0.05, 0.05)  # Hip, thigh, calf

# Absolute range.
MAX_POS_MASS_OFFSET = (0.1, 0.0, 0.05)  # meters
MAX_MASS_OFFSET = 0.5  # kg
MAX_FORCE_DISTURBE = [15, 15, 15]  # N
MAX_TEMPORAL_DURATION = 1  # seconds
DISTURBE_TIME_RANGE = (0, 0.2)  # dsturbe happens only during that time range


class EnvRandomizerMasses(EnvRandomizerBase):
    """
    A randomizer that change the quadruped_gym_env during every reset.
    In particular the masses of the robot links and by adding a mass connected
    to trunk.
    """

    def __init__(
        self,
        env,
        base_mass_err_range=BASE_MASS_ERROR_RANGE,
        leg_mass_err_range=LEG_MASS_ERROR_RANGE,
        max_mass_offset=MAX_MASS_OFFSET,
        max_mass_pos_offset=MAX_POS_MASS_OFFSET,
    ):
        self._env = env
        self._base_mass_err_range = base_mass_err_range
        self._leg_mass_err_range = leg_mass_err_range
        self._max_pos_mass_offset = np.array(max_mass_pos_offset)
        self._max_mass_offset = max_mass_offset
        np.random.seed(0)

    def get_changed_elements(self):
        """Get the env randomizer parameters changed values."""
        self._env.robot._RecordMassAndInertiaInfoFromURDF()
        base_mass = self._env.robot.GetBaseMassFromURDF()
        leg_masses = self._env.robot.GetLegMassesFromURDF()
        feet_masses = self._env.robot.GetFootMassesFromURDF()
        total_mass = np.sum(self._env.robot.GetTotalMassFromURDF())
        offset_mass = self._env.robot.get_offset_mass_value()
        offset_pos = self._env.robot.get_offset_mass_position()
        print("****** Masses info ******")
        print(f"base mass -> {base_mass}")
        print(f"leg masses -> {leg_masses}")
        print(f"feet masses -> {feet_masses}")
        print(f"total mass -> {total_mass:.3f}")
        print(f"offset mass -> {offset_mass}")
        print(f"offset mass position-> {offset_pos}")
        print("*************************")

    def randomize_env(self):
        self._randomize_env()

    def _randomize_env(self):
        self._randomize_base_mass()
        self._randomize_leg_masses()
        self._add_mass_offset()

    def _randomize_base_mass(self):
        robot = self._env.robot
        base_mass = robot.GetBaseMassFromURDF()[0]
        base_mass_low = base_mass * (1.0 + self._base_mass_err_range[0])
        base_mass_high = base_mass * (1.0 + self._base_mass_err_range[1])

        randomized_base_mass = np.random.uniform(base_mass_low, base_mass_high)
        robot.SetBaseMass([randomized_base_mass])

    def _randomize_leg_masses(self):
        """Randomize leg masses. Each leg in the same way."""
        robot = self._env.robot
        leg_masses = robot.GetLegMassesFromURDF()
        leg_masses_lower_bound = np.array(leg_masses) * (1.0 + self._leg_mass_err_range[0])
        leg_masses_upper_bound = np.array(leg_masses) * (1.0 + self._leg_mass_err_range[1])
        randomized_leg_mass = np.random.uniform(leg_masses_lower_bound[0:3], leg_masses_upper_bound[0:3])
        randomized_leg_masses = np.asarray(list(randomized_leg_mass) * robot._robot_config.NUM_LEGS)
        # randomized_leg_masses = [
        #     np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i]) for i in range(len(leg_masses))
        # ]
        robot.SetLegMasses(randomized_leg_masses)

    def _add_mass_offset(self):
        robot = self._env.robot
        base_mass = np.random.uniform(0, self._max_mass_offset)
        block_pos_delta_base_frame = np.random.uniform(-self._max_pos_mass_offset, self._max_pos_mass_offset)
        robot._add_base_mass_offset(base_mass, block_pos_delta_base_frame, is_render=self._env._is_render)


class EnvRandomizerDisturbance(EnvRandomizerBase):
    """
    A randomizer that each simulation step create external forces acting on the robot trunk
    of random intensitiy and temporal duration.
    """

    def __init__(self, env, max_force=MAX_FORCE_DISTURBE, max_time=MAX_TEMPORAL_DURATION, time_range=DISTURBE_TIME_RANGE):
        self._env = env
        self._dt = self._env._sim_time_step
        self._max_force = max_force
        self._max_time = max_time
        self._time_range = time_range

    def randomize_env(self):
        self._create_disturbe()

    def randomize_step(self):
        self._apply_disturbe()

    def _create_disturbe(self):
        force = np.random.uniform(np.zeros(3), np.array(self._max_force))
        temporal_istant = np.random.uniform(self._time_range[0], self._time_range[1])
        duration = np.random.uniform(0, self._max_time)
        self._disturbe = Disturbe(force, temporal_istant, duration)
        self._timer = Timer(dt=self._dt)

    def _apply_disturbe(self):
        robot = self._env.robot
        time = self._env.get_sim_time()
        if time >= self._disturbe._temporal_istant and not self._timer.already_started():
            self._timer.start_timer(timer_time=time, start_time=time, delta_time=self._disturbe._duration)
        cond = self._timer.already_started() and not (robot._is_flying() or self._timer.time_up())
        if cond:
            robot.apply_external_force(self._disturbe._force)
            self._timer.step_timer()


class EnvRandomizerInitialConfiguration(EnvRandomizerBase):
    """Add some noise in the settling robot configuration."""

    def __init__(
        self,
        env,
        max_disturbe=MAX_SETTLING_ACTION_DISTURBANCE,
    ):
        self._env = env
        self._aci = self._env._ac_interface
        self._max_disturbe = np.array(max_disturbe * self._env._robot_config.NUM_LEGS)

    def randomize_env(self):
        # change init config ac_interface
        self._compute_new_init_config()
        # self._aci.set_init_pose(self._new_init_config)

    def randomize_step(self):
        pass

    def _compute_new_init_config(self):
        # sample disturbance and get the new init config
        sample_disturbe = np.random.uniform(np.zeros(self._env._robot_config.NUM_MOTORS), np.array(self._max_disturbe))
        init_action = self._aci._scale_helper_motor_command_to_action(self._aci.get_init_pose())
        new_init_action = init_action + sample_disturbe
        self._new_init_config = self._aci._scale_helper_action_to_motor_command(new_init_action)

    def get_new_init_config(self):
        return self._new_init_config


class Disturbe:
    """
    Class representing an external disturbe object
    """

    def __init__(self, force, temporal_istant, duration):
        self._force = force
        self._temporal_istant = temporal_istant
        self._duration = duration

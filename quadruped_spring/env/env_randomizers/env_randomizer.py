import numpy as np

from quadruped_spring.env.env_randomizers.env_randomizer_base import EnvRandomizerBase

# Relative range.
BASE_MASS_ERROR_RANGE = (-0.1, 0.1)  # 0.2 means 20%
LEG_MASS_ERROR_RANGE = (-0.1, 0.1)  # 0.2 means 20%
MAX_SETTLING_ACTION_DISTURBANCE = (0.1, 0.1, 0.1)  # Hip, thigh, calf

# Values used for nominal springs parameters randomization
SPRING_STIFFNESS_MAX_ERROR_RANGE = (0.1, 0.1, 0.1)  # Hip, thigh, calf
SPRING_DAMPING_MAX_ERROR_RANGE = (0.1, 0.1, 0.1)  # Hip, thigh, calf

# Absolute range.
MAX_POS_MASS_OFFSET = (0.1, 0.0, 0.1)  # meters
MAX_MASS_OFFSET = 1  # kg


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
        super().__init__()

        self._env = env
        self._base_mass_err_range = base_mass_err_range
        self._leg_mass_err_range = leg_mass_err_range
        self._max_pos_mass_offset = np.array(max_mass_pos_offset)
        self._max_mass_offset = max_mass_offset
        self._init()

    def compute_init_total_mass(self):
        self._env.robot._RecordMassAndInertiaInfoFromURDF()
        total_mass = np.sum(self._env.robot.GetTotalMassFromURDF())
        return total_mass

    def _init(self):
        self.total_mass = self.compute_init_total_mass()
        self.base_mass = self._env.robot.GetBaseMassFromURDF()
        self.original_leg_masses = self._env.robot.GetLegMassesFromURDF()
        self.leg_masses = np.sum(self.original_leg_masses)
        self.feet_masses = np.sum(self._env.robot.GetFootMassesFromURDF())
        self.offset_mass = 0

    def randomize_env(self):
        self._randomize_leg_masses()
        self._add_mass_offset()
        self._change_base_mass()

    def _change_base_mass(self):
        robot = self._env.robot
        new_base_mass = self.total_mass - self.offset_mass - self.leg_masses - self.feet_masses
        robot.SetBaseMass([new_base_mass])
        self.base_mass = new_base_mass

    def _randomize_leg_masses(self):
        """Randomize leg masses. Each leg in the same way."""
        robot = self._env.robot
        leg_masses = self.original_leg_masses
        leg_masses_lower_bound = np.array(leg_masses) * (1.0 + self._leg_mass_err_range[0])
        leg_masses_upper_bound = np.array(leg_masses) * (1.0 + self._leg_mass_err_range[1])
        randomized_leg_mass = np.random.uniform(leg_masses_lower_bound[0:3], leg_masses_upper_bound[0:3])
        randomized_leg_masses = np.asarray(list(randomized_leg_mass) * robot._robot_config.NUM_LEGS)
        robot.SetLegMasses(randomized_leg_masses)
        self.leg_masses = np.sum(randomized_leg_masses)

    def _add_mass_offset(self):
        robot = self._env.robot
        base_mass = np.random.uniform(0, self._max_mass_offset)
        block_pos_delta_base_frame = np.random.uniform(-self._max_pos_mass_offset, self._max_pos_mass_offset)
        robot._add_base_mass_offset(base_mass, block_pos_delta_base_frame, is_render=self._env._is_render)
        self.offset_mass = base_mass


class EnvRandomizerSprings(EnvRandomizerBase):
    """Change the springs stiffness and damping coefficients."""

    def __init__(
        self,
        env,
        max_err_stiffness=SPRING_STIFFNESS_MAX_ERROR_RANGE,
        max_err_damping=SPRING_DAMPING_MAX_ERROR_RANGE,
    ):
        super().__init__()

        self._max_err_stiffness = max_err_stiffness
        self._max_err_damping = max_err_damping
        self._env = env
        self.original_stiffness, self.original_damping, _ = self._env.robot.get_spring_nominal_params()

    def randomize_env(self):
        if self._env.are_springs_enabled():
            self._randomize_springs()

    def get_new_stiffness(self):
        stiffness = self.original_stiffness
        stiffness_lower_bound = np.array(stiffness) * (1.0 - np.array(self._max_err_stiffness))
        stiffness_upper_bound = np.array(stiffness) * (1.0 + np.array(self._max_err_stiffness))
        new_stiffness = np.random.uniform(stiffness_lower_bound, stiffness_upper_bound)
        return list(new_stiffness)

    def get_new_damping(self):
        damping = self.original_damping
        damping_lower_bound = np.array(damping) * (1.0 - np.array(self._max_err_damping))
        damping_upper_bound = np.array(damping) * (1.0 + np.array(self._max_err_damping))
        new_damping = np.random.uniform(damping_lower_bound, damping_upper_bound)
        return list(new_damping)

    def _randomize_springs(self):
        self._env.robot.set_spring_stiffness(self.get_new_stiffness())
        self._env.robot.set_spring_damping(self.get_new_damping())


class EnvRandomizerMassesCurriculum(EnvRandomizerBase):
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
        super().__init__()

        self._env = env
        self._curriculum_enabled = True

        self.init_base_mass_err_range = np.array(base_mass_err_range)
        self.init_leg_mass_err_range = np.array(leg_mass_err_range)
        self.init_max_mass_offset = max_mass_offset
        self.init_max_pos_mass_offset = np.array(max_mass_pos_offset)

        self.curriculum_max_mass_offset = 4
        self.curriculum_leg_mass_err_range = np.array([-0.2, 0.2])
        self.curriculum_max_pos_mass_offset = np.array([0.2, 0.0, 0.2])
        self.curriculum_level = 0.0

        self.update_limits_curriculum_based()
        self._init()

    def compute_init_total_mass(self):
        self._env.robot._RecordMassAndInertiaInfoFromURDF()
        total_mass = np.sum(self._env.robot.GetTotalMassFromURDF())
        return total_mass

    def _init(self):
        self.total_mass = self.compute_init_total_mass()
        self.base_mass = self._env.robot.GetBaseMassFromURDF()
        self.original_leg_masses = np.array(self._env.robot.GetLegMassesFromURDF())
        self.leg_masses = np.sum(self.original_leg_masses)
        self.feet_masses = np.sum(self._env.robot.GetFootMassesFromURDF())
        self.offset_mass = 0

    def increase_curriculum_level(self, value):
        new_curr_level = self.curriculum_level + value
        self.curriculum_level = np.clip(new_curr_level, 0, 1)

    def get_curr_value(self, inf, sup):
        return (1 - self.curriculum_level) * inf + self.curriculum_level * sup

    def update_limits_curriculum_based(self):
        self._leg_mass_err_range = self.get_curr_value(self.init_leg_mass_err_range, self.curriculum_leg_mass_err_range)
        self._max_mass_offset = self.get_curr_value(self.init_max_mass_offset, self.curriculum_max_mass_offset)
        self._max_pos_mass_offset = self.get_curr_value(self.init_max_pos_mass_offset, self.curriculum_max_pos_mass_offset)

    def randomize_env(self):
        self.update_limits_curriculum_based()
        self._randomize_env()

    def _randomize_env(self):
        self._randomize_leg_masses()
        self._add_mass_offset()
        self._change_base_mass()

    def _change_base_mass(self):
        robot = self._env.robot
        new_base_mass = self.total_mass - self.offset_mass - self.leg_masses - self.feet_masses
        robot.SetBaseMass([new_base_mass])
        self.base_mass = new_base_mass

    def _randomize_leg_masses(self):
        """Randomize leg masses. Each leg in the same way."""
        robot = self._env.robot
        leg_masses = self.original_leg_masses
        leg_masses_lower_bound = leg_masses * (1.0 + self._leg_mass_err_range[0])
        leg_masses_upper_bound = leg_masses * (1.0 + self._leg_mass_err_range[1])
        randomized_leg_mass = np.random.uniform(leg_masses_lower_bound[0:3], leg_masses_upper_bound[0:3])
        randomized_leg_masses = np.asarray(list(randomized_leg_mass) * robot._robot_config.NUM_LEGS)
        robot.SetLegMasses(randomized_leg_masses)
        self.leg_masses = np.sum(randomized_leg_masses)

    def _add_mass_offset(self):
        robot = self._env.robot
        base_mass = np.random.uniform(0, self._max_mass_offset)
        block_pos_delta_base_frame = np.random.uniform(-self._max_pos_mass_offset, self._max_pos_mass_offset)
        robot._add_base_mass_offset(base_mass, block_pos_delta_base_frame, is_render=self._env._is_render)
        self.offset_mass = base_mass


class EnvRandomizerSpringsCurriculum(EnvRandomizerBase):
    """Change the springs stiffness and damping coefficients."""

    def __init__(
        self,
        env,
        max_err_stiffness=SPRING_STIFFNESS_MAX_ERROR_RANGE,
        max_err_damping=SPRING_DAMPING_MAX_ERROR_RANGE,
    ):
        super().__init__()

        self.init_max_err_stiffness = np.array(max_err_stiffness)
        self.init_max_err_damping = np.array(max_err_damping)

        self.curriculum_max_err_stiffness = np.array([0.3] * 3)
        self.curriculum_max_err_damping = np.array([0.3] * 3)

        self._env = env
        self._curriculum_enabled = True

        original_stiffness, original_damping, _ = self._env.robot.get_spring_nominal_params()

        self.original_stiffness = np.array(original_stiffness)
        self.original_damping = np.array(original_damping)

        self.curriculum_level = 0.0

        self.update_limits_curriculum_based()

    def randomize_env(self):
        if self._env.are_springs_enabled():
            self.update_limits_curriculum_based()
            self._randomize_springs()

    def increase_curriculum_level(self, value):
        new_curr_level = self.curriculum_level + value
        self.curriculum_level = np.clip(new_curr_level, 0, 1)

    def get_curr_value(self, inf, sup):
        return (1 - self.curriculum_level) * inf + self.curriculum_level * sup

    def update_limits_curriculum_based(self):
        self._max_err_stiffness = self.get_curr_value(self.init_max_err_stiffness, self.curriculum_max_err_stiffness)
        self._max_err_damping = self.get_curr_value(self.init_max_err_damping, self.curriculum_max_err_damping)

    def get_new_stiffness(self):
        stiffness_lower_bound = self.original_stiffness * (1.0 - self._max_err_stiffness)
        stiffness_upper_bound = self.original_stiffness * (1.0 + self._max_err_stiffness)
        new_stiffness = np.random.uniform(stiffness_lower_bound, stiffness_upper_bound)
        return new_stiffness

    def get_new_damping(self):
        damping_lower_bound = self.original_damping * (1.0 - self._max_err_damping)
        damping_upper_bound = self.original_damping * (1.0 + self._max_err_damping)
        new_damping = np.random.uniform(damping_lower_bound, damping_upper_bound)
        return new_damping

    def _randomize_springs(self):
        self._env.robot.set_spring_stiffness(self.get_new_stiffness())
        self._env.robot.set_spring_damping(self.get_new_damping())


class EnvRandomizerGround(EnvRandomizerBase):
    """Randomize ground friction coefficient."""

    def __init__(self, env):
        super().__init__()
        self._env = env
        self._mu_min = 0.5

    def randomize_env(self):
        ground_mu_k = self._mu_min + (1 - self._mu_min) * np.random.random()
        self._env._pybullet_client.changeDynamics(self._env.plane, -1, lateralFriction=ground_mu_k)
        if self._env._is_render:
            print("ground friction coefficient is", ground_mu_k)

from email.mime import base
import numpy as np
from quadruped_spring.env.env_randomizers.env_randomizer_base import EnvRandomizerBase


# Relative range.
BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%

class EnvRandomizer(EnvRandomizerBase):
    """A randomizer that change the minitaur_gym_env during every reset."""

    def __init__(self,
                 env,
                 base_mass_err_range=BASE_MASS_ERROR_RANGE,
                 leg_mass_err_range=LEG_MASS_ERROR_RANGE
                 ):
        self._env = env
        self._base_mass_err_range = base_mass_err_range
        self._leg_mass_err_range = leg_mass_err_range
        np.random.seed(0)

    def get_changed_elements(self):
        """Get the env randomizer parameters changed values."""
        base_mass = self._env.robot.GetBaseMassFromURDF()
        leg_masses = self._env.robot.GetLegMassesFromURDF()
        feet_masses = self._env.robot.GetFootMassesFromURDF()
        total_mass = self._env.robot.GetTotalMassFromURDF()
        offset_mass = self._env.robot.get_offset_mass_value()
        offset_pos = self._env.robot.get_offset_mass_position()
        print(f'base mass -> {base_mass}')
        print(f'leg masses -> {leg_masses}')
        print(f'feet masses -> {feet_masses}')
        print(f'total mass -> {total_mass}')
        print(f'offset mass -> {offset_mass}')
        print(f'offset mass position-> {offset_pos}')
        
    def randomize_env(self):
        self._randomize_env()

    def _randomize_env(self):
        robot = self._env.robot
        base_mass = robot.GetBaseMassesFromURDF()
        base_mass_low = base_mass * (1.0 + self._base_mass_err_range[0])
        base_mass_high = base_mass * (1.0 + self._base_mass_err_range[1])
        
        randomized_base_mass = np.random.uniform(base_mass_low, base_mass_high)
        robot.SetBaseMasses(randomized_base_mass)

        leg_masses = robot.GetLegMassesFromURDF()
        leg_masses_lower_bound = np.array(leg_masses) * (
            1.0 + self._leg_mass_err_range[0])
        leg_masses_upper_bound = np.array(leg_masses) * (
            1.0 + self._leg_mass_err_range[1])
        randomized_leg_masses = [
            np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i])
            for i in range(len(leg_masses))
        ]
        robot.SetLegMasses(randomized_leg_masses)

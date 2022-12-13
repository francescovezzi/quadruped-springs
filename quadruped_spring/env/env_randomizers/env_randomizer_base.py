class EnvRandomizerBase:
    """
    Randomizes physical parameters of the objects in the simulation and adds
    perturbations to simulation steps.
    """

    def __init__(self):
        self._env = None
        self._is_curriculum_enabled = False

    def randomize_env(self):
        """Randomize the simulated_objects in the environment.

        Will be called at when env is reset. The physical parameters will be fixed
        for that episode and be randomized again in the next environment.reset().
        Method called before robot is spawned.
        """
        pass

    def randomize_robot(self):
        """Randomize the robot after simulation steps have been applied.

        Will be called at when env is reset. The physical parameters will be fixed
        for that episode and be randomized again in the next environment.reset().
        Method called after robot has been spawned.
        """
        pass

    def randomize_step(self):
        """Randomize simulation steps.

        Will be called at every timestep. May add random forces/torques to Quadruped.
        """
        pass

    def increase_curriculum_level(self, value):
        """Increase the level of randomness."""
        pass


class EnvRandomizerList:
    """Store all the randomizers used."""

    def __init__(self, env_randomizer_list):
        if not isinstance(env_randomizer_list, list):
            raise ValueError("Please use a list of env ranodmizers. Also if it is just one.")
        self._list = env_randomizer_list

    def _init(self, env):
        """Initialize the env randomizers."""
        for idx, env_rand in enumerate(self._list):
            self._list[idx] = env_rand(env)

    def randomize_env(self):
        """Each reset should be called."""
        for env_rand in self._list:
            env_rand.randomize_env()

    def randomize_step(self):
        """Each step should be called."""
        for env_rand in self._list:
            env_rand.randomize_step()

    def randomize_robot(self):
        """
        Each reset after stepping in simulation
        should be called.
        """
        for env_rand in self._list:
            env_rand.randomize_robot()

    def randomize_init(self):
        """Call initialize method after the first reset."""
        for env_rand in self._list:
            env_rand.randomize_init()

    def _set_env(self, env):
        """Update the environment object to randomize wrappers variables."""
        for env_rand in self._list:
            env_rand._env = env

    def is_curriculum_enabled(self):
        for env_rand in self._list:
            if env_rand._is_curriculum_enabled:
                return True
        return False

    def increase_curriculum_level(self, value):
        for env_rand in self._list:
            if env_rand._is_curriculum_enabled:
                env_rand.increase_curriculum_level(value)
        env_rand._env.curriculum_level += value
        print(f"curriculum level set to -> {self._list[0].curriculum_level}")

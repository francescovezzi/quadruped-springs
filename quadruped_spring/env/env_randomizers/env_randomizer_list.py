class EnvRandomizerList:
    """Store all the randomizers used."""

    def __init__(self, env_randomizer_list):
        if not isinstance(env_randomizer_list, list):
            raise ValueError("Please use a list of env ranodmizers. Also if it is just one.")
        self._list = env_randomizer_list

    def _init(self, env):
        """Initialize the env randomizers."""
        self._env = env
        for idx, env_rand in enumerate(self._list):
            self._list[idx] = env_rand(self._env)

    def randomize_env(self):
        """Each reset should be called."""
        for env_rand in self._list:
            env_rand.randomize_env()

    def randomize_step(self):
        """Each step should be called."""
        for env_rand in self._list:
            env_rand.randomize_step()

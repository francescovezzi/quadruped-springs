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

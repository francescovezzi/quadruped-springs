class EnvRandomizerBase:
    """
    Randomizes physical parameters of the objects in the simulation and adds
    perturbations to the stepping of the simulation.
    """

    def randomize_env(self):
        """Randomize the simulated_objects in the environment.

        Will be called at when env is reset. The physical parameters will be fixed
        for that episode and be randomized again in the next environment.reset().
        """
        pass

    def randomize_robot(self):
        """Randomize the robot after simulation steps have been applied.

        Will be called at when env is reset. The physical parameters will be fixed
        for that episode and be randomized again in the next environment.reset().
        """
        pass

    def randomize_step(self):
        """Randomize simulation steps.

        Will be called at every timestep. May add random forces/torques to Minitaur.
        """
        pass

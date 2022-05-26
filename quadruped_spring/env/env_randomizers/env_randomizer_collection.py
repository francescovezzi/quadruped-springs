from quadruped_spring.env.env_randomizers.env_randomizer import (
    EnvRandomizerDisturbance,
    EnvRandomizerInitialConfiguration,
    EnvRandomizerMasses,
    EnvRandomizerSprings,
)
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented observation spaces for deep reinforcement learning:
#   "MASS_RANDOMIZER": For randomly change the mass values of robot links and
#                        adding another to the trunk to vary its COM
#   "DISTURBANCE_RANDOMIZER": For applying random external forces one time for
#                             each episode on the robot trunk
#   "SETTLING_RANDOMIZER": Add some noise in the robot settling configuration
#   "SPRING_RANDOMIZER": Change spring stiffness and dumping


class EnvRandomizerCollection(CollectionBase):
    """Utility to collect all the implemented environment randomizers."""

    def __init__(self):
        super().__init__()
        self._MASS_RANDOMIZER = EnvRandomizerMasses
        self._DISTURBANCE = EnvRandomizerDisturbance
        self._SETTLING = EnvRandomizerInitialConfiguration
        self._SPRINGS = EnvRandomizerSprings
        self._dict = {
            "MASS_RANDOMIZER": [self._MASS_RANDOMIZER],
            "DISTURBANCE_RANDOMIZER": [self._DISTURBANCE],
            "SETTLING_RANDOMIZER": [self._SETTLING],
            "MASS_SETTLING_RANDOMIZER": [self._MASS_RANDOMIZER, self._SETTLING],
            "SPRING_RANDOMIZER": [self._SPRINGS],
            "ALL_RANDOMIZERS": [self._MASS_RANDOMIZER, self._DISTURBANCE, self._SETTLING],
        }
        self._element_type = "env randomizer"

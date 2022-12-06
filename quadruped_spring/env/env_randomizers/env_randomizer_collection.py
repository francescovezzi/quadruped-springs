from quadruped_spring.env.env_randomizers import env_randomizer as er
from quadruped_spring.utils.base_collection import CollectionBase


class EnvRandomizerCollection(CollectionBase):
    """Utility to collect all the implemented environment randomizers."""

    def __init__(self):
        super().__init__()
        self._SPRINGS = er.EnvRandomizerSprings
        self._MASS_RANDOMIZER = er.EnvRandomizerMasses
        self._SPRING_CURRICULUM = er.EnvRandomizerSpringsCurriculum
        self._MASS_CURRICULUM = er.EnvRandomizerMassesCurriculum
        self._GROUND_RANDOMIZER = er.EnvRandomizerGround
        self._dict = {
            "MASS_RANDOMIZER": [self._GROUND_RANDOMIZER, self._MASS_RANDOMIZER],
            "SPRING_RANDOMIZER": [self._GROUND_RANDOMIZER, self._SPRINGS],
            "TEST_RANDOMIZER": [self._GROUND_RANDOMIZER, self._MASS_RANDOMIZER, self._SPRINGS],
            "TEST_RANDOMIZER_CURRICULUM": [self._GROUND_RANDOMIZER, self._MASS_CURRICULUM, self._SPRING_CURRICULUM],
            "GROUND_RANDOMIZER": [self._GROUND_RANDOMIZER],
        }
        self._element_type = "env randomizer"

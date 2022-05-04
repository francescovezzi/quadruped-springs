from quadruped_spring.env.tasks import robot_tasks as rt
from quadruped_spring.utils.base_collection import CollectionBase

# Tasks to be learned with reinforcement learning
#     - "JUMPING_ON_PLACE_HEIGHT"
#         Sparse reward, maximizing max relative height respect the initial one
#         + malus on crashing + bonus on minimizing forward jumping distance
#         + bonus on minimizing the orientation changes respect the default one


class TaskCollection(CollectionBase):
    """Utility to collect all the implemented robot tasks."""

    def __init__(self):
        super().__init__()
        self.JUMPING_ON_PLACE_HEIGHT = rt.JumpingOnPlaceHeight
        self._element_type = "task"
        self._dict = {"JUMPING_ON_PLACE_HEIGHT": self.JUMPING_ON_PLACE_HEIGHT}

from quadruped_spring.env.tasks import robot_tasks as rt
from quadruped_spring.utils.base_collection import CollectionBase

# Tasks to be learned with reinforcement learning


class TaskCollection(CollectionBase):
    """Utility to collect all the implemented robot tasks."""

    def __init__(self):
        super().__init__()
        self.NO_TASK = rt.NoTask
        self._dict = {
            "NO_TASK": self.NO_TASK,
        }
        self._element_type = "task"

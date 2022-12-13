from quadruped_spring.env.tasks.task_base import TaskBase


class NoTask(TaskBase):
    """No tasks is required to be performed. Useful for using TORQUE action interface."""

    def __init__(self, env=None):
        super().__init__(env)

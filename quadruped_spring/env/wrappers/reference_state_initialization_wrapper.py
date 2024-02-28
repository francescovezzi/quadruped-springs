import random

import gym

from quadruped_spring.env.wrappers.get_demonstration_wrapper import GetDemonstrationWrapper as DemoWrapper

TASK_ALLOWED = ["JUMPING_IN_PLACE_DEMO", "JUMPING_FORWARD_DEMO", "BACKFLIP_DEMO"]#, "CONTINUOUS_JUMPING_FORWARD_DEMO"]


class ReferenceStateInitializationWrapper(gym.Wrapper):
    """Wrapper for applying reference state initialization."""

    def __init__(self, env):
        super().__init__(env)
        if self.env.task_env in TASK_ALLOWED:
            self.enable_wrapper = True
            self.demo_list = self.env.task.demo_list
            self.demo_length = self.env.task.demo_length
            self._rng = random.Random()
            self.counter = 0
            self.counter_reset_period = 5
        else:
            self.enable_wrapper = False

    def reset(self):
        if self.enable_wrapper:
            self.random_el = self.compute_random_el()
            # print(f'random el: {self.random_el}')
            # print(f"robot initialized at element {self.random_el}/{self.demo_length - 1} of the demonstration")
            random_demo = DemoWrapper.read_demo(self.demo_list[self.random_el])
            self.env.set_robot_desired_state(random_demo)
            self.env.task.set_demo_counter(value=self.random_el)
        return super().reset()

    def compute_random_el(self):
        limit = self.demo_length - 5
        # print(f'limit: {limit}')
        if self.counter == self.counter_reset_period:
            self.counter = 0
            limit = self.demo_length // 5
        else:
            self.counter += 1
        return self._rng.randint(0, limit - 1)

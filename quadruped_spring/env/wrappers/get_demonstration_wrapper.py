import os

import gym
import numpy as np


class GetDemonstrationWrapper(gym.Wrapper):
    """Wrapper used for collecting demonstrations."""

    def __init__(self, env, path):
        super().__init__(env)
        self.id = 0
        self.path = path
        self.env.wrapped_demo_env = self  # For the GoToRestWrapper

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        demo = self._get_demo()
        self.demo_list.append(demo)

        return obs, reward, done, infos

    def reset(self):
        self.landing_started = False
        self.demo_list = []
        return super().reset()

    def save_demo(self):
        name = f"demo_list_{self.id}.npy"
        demo_list = self.demo_list[0:-1]
        np.save(os.path.join(self.path, name), demo_list)
        print(f"demo of shape {np.shape(demo_list)} saved in {self.path}/{name}")
        self.id += 1

    def _get_demo(self):
        r = self.env.robot
        action = self.env.get_last_filtered_action()
        joint_position = r.GetMotorAngles()
        joint_velocity = r.GetMotorVelocities()
        base_position = r.GetBasePosition()
        base_linear_velocity = r.GetBaseLinearVelocity()
        base_orientation = r.GetBaseOrientation()  # Quaternion
        base_angular_velocity = r.GetBaseAngularVelocity()
        if not self.landing_started:
            if self.env.task.is_switched_controller() and base_linear_velocity[2] <= 0.0:
                self.landing_started = True
        return np.concatenate(
            (
                action,
                joint_position,
                joint_velocity,
                base_position,
                base_orientation,
                base_linear_velocity,
                base_angular_velocity,
                [self.landing_started],
            )
        )

    @staticmethod
    def read_demo(demo, action_dim=6, num_joints=12):
        ret = []
        intervals = np.cumsum(np.array([action_dim, num_joints, num_joints, 3, 4, 3, 3, 1]))
        first_idx = 0
        for last_idx in intervals:
            tmp = demo[first_idx:last_idx]
            first_idx = last_idx
            ret.append(tmp)

        return ret

import inspect
import os

import gym
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)

from env.quadruped_gym_env import QuadrupedGymEnv
from env.wrappers.landing_wrapper import LandingWrapper


class JumpingStateMachine(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.aci = self.env.get_ac_interface()
        self.j_front = 0.0
        self.j_rear = 0.0
        self.init_pose_front = self.aci.get_settling_pose()[:6]
        self.init_pose_rear = self.aci.get_settling_pose()[6:]
        self.init_pose = self.aci.get_settling_pose()
        self.init_action_front = self.aci._transform_motor_command_to_action(self.init_pose)[:6]
        self.init_action_rear = self.aci._transform_motor_command_to_action(self.init_pose)[6:]
        self.scale = 0.9
        x = self.init_action_front[0]
        y = -self.init_action_front[1]
        zet = -self.scale * 1
        self.final_action_front = np.array([x, -y, zet, x, y, zet])
        self.final_action_rear = np.array([x, -y, zet, x, y, zet])

    @staticmethod
    def _interpolate_action(a, b, p):
        p = np.clip(p, 0, 1)
        return a * (1 - p) + b * p

    def compute_front_action(self):
        return self._interpolate_action(self.init_action_front, self.final_action_front, self.j_front)

    def compute_rear_action(self):
        return self._interpolate_action(self.init_action_rear, self.final_action_rear, self.j_rear)

    def jumping_explosive_action(self):
        if self.env.are_springs_enabled():
            coeff = 0.7
        else:
            coeff = 1.0
        action_front = self.compute_front_action()
        action_rear = self.compute_rear_action()
        self.j_front += 0.0
        self.j_rear += 0.1
        jump_action = np.concatenate((action_front, action_rear))
        return jump_action

    def reset(self):
        obs = self.env.reset()
        self.aci._load_springs(1.0)
        self.j_rear = 0.8
        self.j_front = 0.8
        return obs


def build_env():
    env_config = {
        "enable_springs": True,
        "render": True,
        "on_rack": False,
        "isRLGymInterface": True,
        "motor_control_mode": "CARTESIAN_PD",
        "action_repeat": 10,
        "record_video": False,
        "action_space_mode": "DEFAULT",
        "task_env": "JUMPING_IN_PLACE",
        "env_randomizer_mode": "GROUND_RANDOMIZER",
        "curriculum_level": 1.0,
        "observation_space_mode": "ARS_BASIC",
    }
    env = QuadrupedGymEnv(**env_config)
    env = JumpingStateMachine(env)
    env = LandingWrapper(env)
    return env


if __name__ == "__main__":

    env = build_env()
    obs = env.reset()

    done = False
    while not done:
        action = env.jumping_explosive_action()
        obs, reward, done, info = env.step(action)
    success = info["TimeLimit.truncated"]

    env.close()
    print("end")

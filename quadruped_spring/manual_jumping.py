import inspect
import os
import time

import gym
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)

import argparse

from env.quadruped_gym_env import QuadrupedGymEnv
from env.wrappers.landing_wrapper import LandingWrapper
from env.wrappers.rest_wrapper import RestWrapper
from utils.evaluate_metric import EvaluateMetricJumpOnPlace

# from utils.monitor_state import MonitorState


class JumpingStateMachine(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def jumping_explosive_action(self):
        if self.env._enable_springs:
            coeff = 0.2
        else:
            coeff = 0.2
        action_front = np.array([0, 0, coeff * 1] * 2)
        action_rear = np.array([0, 0, 1] * 2)
        jump_action = np.concatenate((action_front, action_rear))
        return jump_action

    def step(self, action):

        obs, reward, done, infos = self.env.step(action)

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()


def build_env(enable_springs=False):
    env_config = {
        "enable_springs": enable_springs,
        "render": True,
        "on_rack": False,
        "isRLGymInterface": True,
        "motor_control_mode": "PD",
        "action_repeat": 10,
        "record_video": False,
        "action_space_mode": "DEFAULT",
        "task_env": "JUMPING_FORWARD",
        "enable_env_randomization": False,
        "env_randomizer_mode": "DISTURBANCE_RANDOMIZER",
        "preload_springs": True,
    }
    env_config["enable_springs"] = True
    if fill_line:
        env_config["render"] = False
    env = QuadrupedGymEnv(**env_config)
    return env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-springs", action="store_true", default=False, help="enable springs")
    parser.add_argument("--fill-line", action="store_true", default=False, help="fill line in report.txt")
    args = parser.parse_args()
    enable_springs = args.enable_springs
    fill_line = args.fill_line

    env = build_env(enable_springs=enable_springs)
    env = JumpingStateMachine(env)
    # sim_steps = env._total_sim_steps + 3000
    # env = MonitorState(env=env, path="logs/plots/manual_jumping_with_springs", rec_length=sim_steps)
    env = EvaluateMetricJumpOnPlace(env)
    env = RestWrapper(env)
    env = LandingWrapper(env)
    done = False
    obs = env.reset()
    t_start = time.time()
    while not done:
        action = env.jumping_explosive_action()
        # action = np.zeros(12)
        obs, reward, done, info = env.step(action)
    t_end = time.time() - t_start

    # env.release_plots()
    print("******")
    print(f"simulation in reality lasted for -> {t_end}")
    print(f"simulation in simulation lasted for -> {env.get_sim_time()}")
    print(f"reward -> {reward}")
    print(f"min_height -> {env.get_metric().height_min}")
    print("******")
    if fill_line:
        report_path = os.path.join(current_dir, "logs", "models", "performance_report.txt")
        with open(report_path, "w") as f:
            f.write(env.print_first_line_table())
            if enable_springs:
                f.write(env.fill_line(id="ManualWithSprings"))
            else:
                f.write(env.fill_line(id="ManualWithoutSprings"))
    else:
        env.print_metric()
    env.close()
    print("end")

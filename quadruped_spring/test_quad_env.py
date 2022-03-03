import os, sys, inspect, pathlib
current_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, root_dir)

import numpy as np

from env.quadruped_gym_env import QuadrupedGymEnv

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

# stable baselines
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env


from gym.wrappers.time_limit import TimeLimit

env = QuadrupedGymEnv(render=True, enable_springs=True)
env = TimeLimit(TimeLimit(env, max_episode_steps=3000))
env = make_vec_env(env_id="QuadrupedSpring-v0", n_envs=1, seed=24)
print(env.get_attr('robot')[0])
# for i in range(1000):
#     action = np.full(12, -1)
#     env.step(action)
# env.close()
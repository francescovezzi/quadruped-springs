import inspect
import os
import re
import sys
import time
from sys import platform

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

# stable baselines
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.file_utils import get_latest_directory, get_latest_model, load_all_results

# utils
from utils.utils import plot_results

LEARNING_ALG = "PPO"
interm_dir = "logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '111121133812'
log_dir = get_latest_directory(interm_dir)
# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config["render"] = False
env_config["record_video"] = False
env_config["add_noise"] = False
env_config["motor_control_mode"] = "CARTESIAN_PD"
env_config["observation_space_mode"] = "LR_COURSE_OBS"
env_config["test_env"] = False

# get latest model and normalization stats, and plot
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
# monitor_results = load_results(log_dir)
# print(monitor_results)
# plot_results([log_dir], 10e10, "timesteps", LEARNING_ALG + " ")
# plt.show()

# reconstruct env
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
# env = VecVideoRecorder(env, video_folder=log_dir, video_length=2000, record_video_trigger=lambda x: x==0)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation
#

for i in range(5000):
    action, _states = model.predict(obs, deterministic=True)  # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print("episode_reward", episode_reward)
        episode_reward = 0

    # [TODO] save data from current robot states for plots

# [TODO] make plots:


print("end of load_sb3.py")

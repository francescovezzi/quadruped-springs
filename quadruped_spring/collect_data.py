import inspect
import os
import uuid

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from importlib import import_module

import yaml
from sb3_contrib import ARS
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.quadruped_gym_env import QuadrupedGymEnv

LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
ENV_ID = "QuadrupedSpring-v0"
ID = "5"


def callable_env(env_id, wrappers, kwargs):
    def aux():
        env = env_id(**kwargs)
        for wrapper in wrappers:
            module = ".".join(wrapper.split(".")[:-1])
            class_name = wrapper.split(".")[-1]
            module = import_module(module)
            wrap = getattr(module, class_name)
            env = wrap(env)
        return env

    return aux


# define directories
aux_dir = "logs/models"
model_dir = os.path.join(currentdir, aux_dir, LEARNING_ALG, f"{ENV_ID}_{ID}")
model_file = os.path.join(model_dir, "best_model.zip")
args_file = os.path.join(model_dir, ENV_ID, "args.yml")
stats_file = os.path.join(model_dir, ENV_ID, "vecnormalize.pkl")
data_dir = os.path.join(model_dir, f"data_{str(uuid.uuid4())}")
os.makedirs(data_dir, exist_ok=True)

# Load env kwargs
env_kwargs = {}
if os.path.isfile(args_file):
    with open(args_file, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]

wrapper_list = loaded_args["hyperparams"]["env_wrapper"]

# build env
env_kwargs["enable_env_randomization"] = False
env_kwargs["env_randomizer_mode"] = "SETTLING_RANDOMIZER"
env = callable_env(QuadrupedGymEnv, wrapper_list, env_kwargs)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_file, env)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
model = LEARNING_ALGS[LEARNING_ALG].load(model_file, env)
obs_list = []
action_list = []
n_episodes = 1
obs = env.reset()

for _ in range(n_episodes):
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0, :])
        action_list.append(action[0, :])
        obs, rewards, done, info = env.step(action)

obs_array = np.array(obs_list)
action_array = np.array(action_list)

np.save(data_dir + "/obs.npy", obs_array)
np.save(data_dir + "/actions.npy", action_array)

env.close()
print("end")

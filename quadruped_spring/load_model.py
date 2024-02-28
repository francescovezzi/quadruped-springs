import glob
import inspect
import os

import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 12})
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
rl_zoo_dir = os.path.join(os.path.dirname(os.path.dirname(currentdir)), "rl-baselines3-zoo")

from importlib import import_module

import yaml
from sb3_contrib import ARS
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from quadruped_spring.env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.env.wrappers.go_to_rest_wrapper import GoToRestWrapper
from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper
from quadruped_spring.env.wrappers.landing_wrapper_2 import LandingWrapper2
from quadruped_spring.env.wrappers.landing_wrapper_backflip import LandingWrapperBackflip
from quadruped_spring.env.wrappers.landing_wrapper_continuous import LandingWrapperContinuous
from quadruped_spring.env.wrappers.landing_wrapper_continuous2 import LandingWrapperContinuous2
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.utils.monitor_state import MonitorState
from quadruped_spring.scripts.mathematica_wrapper import MathematicaWrapper

SEED = 2409

LEARNING_ALGS = {"ars": ARS, "ppo": PPO}

# TASK = "CONTINUOUS_JUMPING_FORWARD_PPO"
TASK = "CONTINUOUS_JUMPING_FORWARD3"
# TASK = "BACKFLIP_PPO"
ALGO = "ppo"
ENV_NAME = "QuadrupedSpring-v0"
MODEL = "best_model.zip"
RENDER = False

TRAINING_STEPS = ["ars", "demo_ppo", "retrained_ppo"]
TRAINING_STEP_INDEX = 2
TRAINING_STEP = TRAINING_STEPS[TRAINING_STEP_INDEX]

# MATHEMATICA_PATH = os.path.join(currentdir, "raw-data", "jumping_in_place_2stages_comparisons")
# MATHEMATICA_PATH = os.path.join(currentdir, "raw-data", "jumping_forward_2stages_comparisons")
MATHEMATICA_PATH = os.path.join(currentdir, "raw-data/v2", "jumping_forward_rigid")

# _source_path = os.path.join("logs_final_cmp", ALGO, "jumping_forward-10_27", "springs")
# _source_path = os.path.join("f_logs_ppo_comp", "jumping_forward-10_27_b", "demo_ppo")
# _source_path = os.path.join("f_logs_ppo_comp", "jumping_forward-10_27_c", TRAINING_STEP)
# _source_path = os.path.join("f_logs_ppo_comp", "jumping_in_place-10_28", TRAINING_STEP)
_source_path = os.path.join("logs_final_cmp", ALGO, "jumping_forward-10_27", "no_springs")
source_path = os.path.join(currentdir, "logs", _source_path)
# _source_path = os.path.join("log_backflip", ALGO, 'QuadrupedSpring-v0_3')
# _source_path = os.path.join("logs_continuous_jf", ALGO, "QuadrupedSpring-v0_19")
# _source_path = os.path.join("logs_continuous_jf", ALGO, "QuadrupedSpring-v0_10")
# _source_path = os.path.join("logs_continuous_jf", ALGO, "QuadrupedSpring-v0_17")
# _source_path = os.path.join("logs_continuous_jf_2", ALGO, "QuadrupedSpring-v0_5")
# source_path = os.path.join(rl_zoo_dir, "logs", _source_path)

stats_path = os.path.join(source_path, f"{ENV_NAME}/vecnormalize.pkl")
model_path = os.path.join(source_path, MODEL)


def load_env_kwargs(src):
    args_path = os.path.join(src, f"{ENV_NAME}/args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            return loaded_args
    else:
        raise RuntimeError(f"{args_path} file not found.")


def get_env_kwargs(src):
    env_kwargs = {}
    loaded_args = load_env_kwargs(src)
    if loaded_args["env_kwargs"] is not None:
        env_kwargs = loaded_args["env_kwargs"]
        env_kwargs["render"] = RENDER
        env_kwargs["env_randomizer_mode"] = "GROUND_RANDOMIZER"
        env_kwargs["task_env"] = TASK
        adapt_args(env_kwargs)
    return env_kwargs


def callable_env(kwargs):
    def aux():
        env = QuadrupedGymEnv(**kwargs)
        env = MathematicaWrapper(env, output_path=MATHEMATICA_PATH, algo=TRAINING_STEP)
        # env = LandingWrapperContinuous2(env)
        env = GoToRestWrapper(env)
        # env = LandingWrapperBackflip(env)
        # env = LandingWrapper2(env)
        # env = LandingWrapper(env)
        env = ObsFlatteningWrapper(env)
        return env

    return aux


def adapt_args(kwargs):
    to_eliminate = ["add_noise", "enable_env_randomization", "aux_seed"]
    for e in to_eliminate:
        if e in kwargs.keys():
            del kwargs[e]


if __name__ == "__main__":

    env_kwargs = get_env_kwargs(source_path)
    env = callable_env(env_kwargs)
    env = make_vec_env(env, n_envs=1)
    env = VecNormalize.load(stats_path, env)
    env.training = False  # do not update stats at test time
    env.norm_reward = False  # reward normalization is not needed at test time

    # load model
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = LEARNING_ALGS[ALGO].load(model_path, env, custom_objects=custom_objects)
    set_random_seed(SEED)

    sim_steps = 1500
    obs = env.reset()
    done = False
    rew = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rew += reward[0]

    print(f"rew: {rew}")
    env.close()
    print("end")

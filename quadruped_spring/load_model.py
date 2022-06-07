import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from importlib import import_module

import yaml
from sb3_contrib import ARS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import set_random_seed

from env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.env.wrappers.rest_wrapper import RestWrapper
from quadruped_spring.utils.monitor_state import MonitorState
from quadruped_spring.utils.video_recording import VideoRec


SEED = 24

LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
ENV_ID = "QuadrupedSpring-v0"
ID = "1"

REC_VIDEO = False
SAVE_PLOTS = False
RENDER = False
EVAL_EPISODES = 10
ENABLE_ENV_RANDOMIZATION = False
ENV_RANDOMIZER = "MASS_SETTLING_RANDOMIZER"


def callable_env(env_id, wrappers, kwargs):
    def aux():
        env = env_id(**kwargs)
        env = RestWrapper(env)
        if SAVE_PLOTS:
            plot_folder = f"logs/plots/{LEARNING_ALG}_{ENV_ID}_{ID}"
            env = MonitorState(env, path=plot_folder)
        if REC_VIDEO:
            video_folder = "logs/videos/"
            video_name = f"{LEARNING_ALG}_{ENV_ID}_{ID}"
            env = VideoRec(env, video_folder, video_name)
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
model_file = os.path.join(model_dir, "rl_model_12000000_steps")
args_file = os.path.join(model_dir, ENV_ID, "args.yml")
stats_file = os.path.join(model_dir, ENV_ID, "vecnormalize.pkl")

# Load env kwargs
env_kwargs = {}
if os.path.isfile(args_file):
    with open(args_file, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]
if RENDER:
    env_kwargs["render"] = True
wrapper_list = loaded_args["hyperparams"]["env_wrapper"]

# build env
env_kwargs["enable_env_randomization"] = ENABLE_ENV_RANDOMIZATION
env_kwargs["env_randomizer_mode"] = ENV_RANDOMIZER
env_kwargs["curriculum_level"] = 0.95
env = callable_env(QuadrupedGymEnv, wrapper_list, env_kwargs)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_file, env)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
model = LEARNING_ALGS[LEARNING_ALG].load(model_file, env, custom_objects=custom_objects)
print(f"\nLoaded model: {model_file}\n")

set_random_seed(SEED)
obs = env.reset()
n_episodes = EVAL_EPISODES
total_reward = 0
total_success = 0
for _ in range(n_episodes):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    total_reward += rewards[0]
    total_success += info[0]['TimeLimit.truncated']

avg_reward = total_reward / n_episodes
avg_success = total_success / n_episodes

if REC_VIDEO:
    env.env_method("release_video", indices=0)
if SAVE_PLOTS:
    env.env_method("release_plots", indices=0)

env_randomizer = env.env_method("get_randomizer_mode", indices=0)[0]
print('\n')
print(f'over {n_episodes} episodes using {env_randomizer} env randomizer:')
print(f'average reward -> {avg_reward}' )
print(f'average success -> {avg_success}')
print('\n')

env.close()
print("end")

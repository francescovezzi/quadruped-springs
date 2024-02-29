import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
quadruped_spring_dir = currentdir
currentdir_ = os.path.dirname(currentdir)
currentdir_ = os.path.dirname(currentdir_)
rl_zoo_dir = os.path.join(currentdir_, "rl-baselines3-zoo")

import yaml
from sb3_contrib import ARS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from env.quadruped_gym_env import QuadrupedGymEnv
from env.wrappers.landing_wrapper_2 import LandingWrapper2
from quadruped_spring.env.wrappers.get_demonstration_wrapper import GetDemonstrationWrapper
from quadruped_spring.env.wrappers.go_to_rest_wrapper import GoToRestWrapper
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.env.wrappers.save_demo_wrapper import SaveDemoWrapper

SEED = 24

LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
ENV_ID = "QuadrupedSpring-v0"
MODEL = "best_model"

REC_VIDEO = False
SAVE_PLOTS = False
RENDER = False

LOG_DIR = os.path.join(rl_zoo_dir, "logs", "ars")
ID = 19


def callable_env(env_id, kwargs):
    def aux():
        env = env_id(**kwargs)
        env = GetDemonstrationWrapper(env, path=os.path.join(currentdir, "demonstrations"))
        env = LandingWrapper2(env)
        env = GoToRestWrapper(env)
        env = ObsFlatteningWrapper(env)
        env = SaveDemoWrapper(env)
        return env

    return aux

# define directories
model_dir = os.path.join(LOG_DIR, f"{ENV_ID}_{ID}")
model_file = os.path.join(model_dir, MODEL)
args_file = os.path.join(model_dir, ENV_ID, "args.yml")
stats_file = os.path.join(model_dir, ENV_ID, "vecnormalize.pkl")


# Load env kwargs
env_kwargs = {}
if os.path.isfile(args_file):
    with open(args_file, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]
            env_kwargs["verbose"] = 1
            env_kwargs["render"] = RENDER

# build env
env = callable_env(QuadrupedGymEnv, env_kwargs)
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

# run model
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
count = 0

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)

env.close()
print("end")

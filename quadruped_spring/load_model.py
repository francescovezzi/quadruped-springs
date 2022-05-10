import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from importlib import import_module
import yaml
from sb3_contrib import ARS
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.utils.evaluate_metric import EvaluateMetricJumpOnPlace
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper

LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
ENV_ID = "QuadrupedSpring-v0"
ID = "4"

def callable_env(env_id, wrappers, kwargs):
    def aux():
        env = env_id(**kwargs)
        for wrapper in wrappers:
            module = ".".join(wrapper.split(".")[:-1])
            class_name = wrapper.split(".")[-1]
            module = import_module(module)
            wrap = getattr(module, class_name)
            env = wrap(env)
        env = EvaluateMetricJumpOnPlace(env)
        return env
    return aux

# define directories
aux_dir = "logs/models"
model_dir = os.path.join(currentdir, aux_dir, LEARNING_ALG, f"{ENV_ID}_{ID}")
model_file = os.path.join(model_dir, "best_model.zip")
args_file = os.path.join(model_dir, ENV_ID, "args.yml")
stats_file = os.path.join(model_dir, ENV_ID, "vecnormalize.pkl")

# Load env kwargs
env_kwargs = {}
if os.path.isfile(args_file):
    with open(args_file, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]
env_kwargs["render"] = True
wrapper_list = loaded_args["hyperparams"]['env_wrapper']

# build env
env = callable_env(QuadrupedGymEnv, wrapper_list, env_kwargs)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_file, env)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
model = LEARNING_ALGS[LEARNING_ALG].load(model_file, env)
print(f"\nLoaded model: {model_file}\n")

sim_steps = 1200
obs = env.reset()
episode_reward = 0
for i in range(sim_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print(f"episode_reward: {episode_reward}")
        episode_reward = 0
        break
        # obs = env.reset()
env.env_method("print_metric", indices=0)
env.close()
print("end")

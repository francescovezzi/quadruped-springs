import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import yaml
from env.quadruped_gym_env import QuadrupedGymEnv

from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import ARS


LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
ENV_ID = "QuadrupedSpring-v0"
ID = 26


#define directories
aux_dir = "logs/models"
model_dir = os.path.join(currentdir, aux_dir, LEARNING_ALG, f"{ENV_ID}_{ID}")
model_file = os.path.join(model_dir, 'best_model.zip')
args_file = os.path.join(model_dir, ENV_ID, 'args.yml')
stats_file = os.path.join(model_dir, ENV_ID, 'vecnormalize.pkl')

# Load env kwargs
env_kwargs = {}
if os.path.isfile(args_file):
    with open(args_file, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]
env_kwargs['render'] = True
if ID == 26:
    env_kwargs['adapt_spring_parameters'] = False

# build env
env = lambda: QuadrupedGymEnv(**env_kwargs)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_file, env)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
model = LEARNING_ALGS[LEARNING_ALG].load(model_file, env)
print(f"\nLoaded model: {model_file}\n")

sim_steps = 2000
obs = env.reset()
episode_reward = 0
for i in range(sim_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print(f"episode_reward: {episode_reward}")
        episode_reward = 0

env.close()
print("end")
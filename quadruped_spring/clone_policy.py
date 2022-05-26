import glob
import inspect
import os

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

LEARNING_ALG = "ars"
ENV_ID = "QuadrupedSpring-v0"
ID = "5"

# Data folders
aux_dir = "logs/models"
model_dir = os.path.join(currentdir, aux_dir, LEARNING_ALG, f"{ENV_ID}_{ID}")
data_dirs = []
for f in glob.glob(os.path.join(model_dir, "data_*")):
    _f = f.split("/")[-1]
    if _f.split("_")[0] == "data":
        data_dirs.append(f)

# Get Data
obs_list = []
action_list = []
for data_dir in data_dirs:
    obs_list.append(np.load(os.path.join(data_dir, "obs.npy")))
    action_list.append(np.load(os.path.join(data_dir, "actions.npy")))
total_obs = np.concatenate(tuple(obs_list), axis=0)
total_actions = np.concatenate(tuple(action_list), axis=0)

print(np.shape(total_obs))
print(np.shape(obs_list[0]))

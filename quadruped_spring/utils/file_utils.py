""" file management, copying etc. """
import os
import inspect 
import json
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


from stable_baselines3.common.monitor import load_results
from .utils import plot_curves, ts2xy


def write_env_config(destination_directory, vec_env, updated_config=None):
	"""Write configurations to file. """
	try:
		signature = inspect.getargspec(vec_env.venv.envs[0].env.__init__)
	except:
		# not using vec env
		signature = inspect.getargspec(vec_env.__init__)
	#print('signature', signature)
	#print('config', os.path.join(destination_directory,"env_configs.txt"))
	with open(os.path.join(destination_directory,"env_configs.txt"),"w") as outfile:
		args_dict = {}
		# skip urdf root and config file (should be obvious from context) for json
		for i,k,v in zip(range(len(signature.defaults)),signature.args[1:],signature.defaults):
			#print(k,v)
			outfile.write( str(k) + ' ' + str(v) +'\n')
			if i>2:
				args_dict[k] = v

	if updated_config:
		args_dict.update(updated_config)
	#print(args_dict)
	#print('config json', os.path.join(destination_directory,'env_configs.json'))
	with open(os.path.join(destination_directory,'env_configs.json'), "w") as outfile:
		json.dump(args_dict,outfile)

##########################################################################################################
# Stable baselines 
##########################################################################################################
def get_latest_model(path):
	""" Returns most recent model saved in path directory. """
	files = os.listdir(path)
	paths = [os.path.join(path, basename) for basename in files if basename.endswith('.zip')]
	return max(paths, key=os.path.getctime)

def read_env_config(directory):
	"""Read environment configuration from directory. """
	with open(os.path.join(directory,'env_configs.json')) as f:
		env_config = json.load(f)
	return env_config

# stable baselines 3
def get_sorted_dirs(path):
	files = os.listdir(path)
	print('files', files)
	dirs = [os.path.join(path,basename) for basename in files if os.path.isdir(os.path.join(path,basename))]
	print('dirs', dirs)
	# and basename is not '__pycache__'
	if '__pycache__' in dirs:
		dirs.remove('__pycache__')
	return sorted(dirs,key=os.path.getctime)

def load_all_results(path):
	"""Read all monitor files recursively, concatenate data together, plot. """
	tslist = []
	xaxis = 'timesteps'
	# get in order 0... N gas dirs
	gas_dirs = get_sorted_dirs(path)
	print('gas dirs', gas_dirs)
	# each gas dir may have multiple directories w monitor files, depending on performance
	for gas_dir in gas_dirs:
		trial_dirs = get_sorted_dirs(gas_dir)

		print('trial dirs', trial_dirs)

		# extract monitor from each
		for trial_dir in trial_dirs:
			#for folder in dirs:
			timesteps = load_results(trial_dir)
			#if num_timesteps is not None:
			timesteps = timesteps[timesteps.l.cumsum() <= 10e10]
			tslist.append(timesteps)

	# iterate through and concatenate data 
	xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
	xy_list = concatenate_xy(xy_list)
	plot_curves(xy_list,xaxis, 'A1 Ep Rewards')
	plt.ylabel("Episode Rewards")

	xy_list = [ts2xy(timesteps_item, xaxis, True) for timesteps_item in tslist]
	xy_list = concatenate_xy(xy_list)
	plot_curves(xy_list, xaxis, 'A1 Ep Len')
	plt.ylabel("Episode Length")

def concatenate_xy(xy_list):
	curr_t = xy_list[0][0]
	curr_x = xy_list[0][1]
	new_xy = [np.array([0]),np.array([0])]
	for xy in xy_list:
		new_xy[0] = np.concatenate((np.array(new_xy[0]), np.array(xy[0])+new_xy[0][-1]))
		new_xy[1] = np.concatenate((np.array(new_xy[1]), xy[1]))

	return [tuple(new_xy)]

##########################################################################################################
# rllib
##########################################################################################################
def get_latest_directory(path):
	""" Returns most recent directory in path. """
	files = os.listdir(path)
	paths = [os.path.join(path, f) for f in files if os.path.isdir(os.path.join(path, f))]
	paths = [path for path in paths if '__pycache__' not in path]
	return max(paths, key=os.path.getctime)

def get_latest_model_rllib(path):
	""" Returns most recent model saved in path directory. """
	checkpoint = get_latest_directory(path)
	files = os.listdir(checkpoint)
	paths = [os.path.join(checkpoint, file) for file in files if ( file.startswith('checkpoint') and not file.endswith('.tune_metadata') )]
	print(paths)
	return paths[0]
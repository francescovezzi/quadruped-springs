# Quadruped-Sim

This repository contains an environment for simulating a quadruped robot accomplishing several higly-dynamical jumping based tasks
such as:
- explosive jumping in place and forward
- continuous jumping forward
- backflip

## Installation

Recommend using conda with python3.7 or higher. After instaalling conda, this can be done as follows:

`conda create -n [YOUR_ENV_NAME] python=3.7`

To activate the virtualenv: 

`conda activate [YOUR_ENV_NAME]` 

Install all dependencies:

`pip install -r requirements.txt `

## Code structure

- [env](./env) for the quadruped environment files, please see the gym simulation environment [quadruped_gym_env.py](./env/quadruped_gym_env.py), the robot specific functionalities in [quadruped.py](./env/quadruped.py), and config variables in [configs_go1.py](./env/configs_go1.py). You will need to make edits in [quadruped_gym_env.py](./env/quadruped_gym_env.py), and review [quadruped.py](./env/quadruped.py) carefully for accessing robot states and calling functions to solve inverse kinematics, return the leg Jacobian, etc. 
- [go1_description](./go1_description) contains the robot mesh files and urdf.
- [utils](./utils) for some file i/o and plotting helpers.
- [hopf_network.py](./hopf_polar.py) provides a CPG class skeleton for various gaits, and maps these to be executed on an instance of the  [quadruped_gym_env](./env/quadruped_gym_env.py) class. Please fill in this file carefully. 
- [load_sb3.py](./load_sb3.py) provide an interface for loading a RL pre-trained model based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). For training take a look at the documentation or at [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The quadruped environment took inspiration from [Google's motion-imitation repository](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf). 
- Reinforcement learning algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Also see for example [ray[rllib]](https://github.com/ray-project/ray) and [spinningup](https://github.com/openai/spinningup). 

## Acknowledgements

The quadruped env was original created by [Guillaume Bellegarda](https://scholar.google.com/citations?user=YDimn5wAAAAJ&hl=en)
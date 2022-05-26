# Quadruped-Sim
This repository contains an environment for simulating a quadruped robot.

## Installation

Recommend using a virtualenv (or conda) with python3.6 or higher. After installing virtualenv with pip, this can be done as follows:

`virtualenv {quad_env, or choose another name venv_name} --python=python3`

To activate the virtualenv: 

`source {PATH_TO_VENV}/bin/activate` 

Your command prompt should now look like: 

`(venv_name) user@pc:path$`

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

## Conceptual resources
The CPGs are based on the following papers:
- L. Righetti and A. J. Ijspeert, "Pattern generators with sensory feedback for the control of quadruped locomotion," 2008 IEEE International Conference on Robotics and Automation, 2008, pp. 819-824, doi: 10.1109/ROBOT.2008.4543306. [link](https://ieeexplore.ieee.org/document/4543306)
- M. Ajallooeian, S. Pouya, A. Sproewitz and A. J. Ijspeert, "Central Pattern Generators augmented with virtual model control for quadruped rough terrain locomotion," 2013 IEEE International Conference on Robotics and Automation, 2013, pp. 3321-3328, doi: 10.1109/ICRA.2013.6631040. [link](https://ieeexplore.ieee.org/abstract/document/6631040) 
- M. Ajallooeian, S. Gay, A. Tuleu, A. Spröwitz and A. J. Ijspeert, "Modular control of limit cycle locomotion over unperceived rough terrain," 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2013, pp. 3390-3397, doi: 10.1109/IROS.2013.6696839. [link](https://ieeexplore.ieee.org/abstract/document/6696839) 

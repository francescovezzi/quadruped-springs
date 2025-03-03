# Quadruped-Sim

This repository contains an environment for simulating a quadruped robot optionally equipped with PEA (Parallel Elastic Actuator) accomplishing several higly-dynamical jumping based tasks as:

- explosive jumping in place and forward
- continuous jumping forward
- backflip

using the methodology described here: [Two-Stage Learning](https://arxiv.org/pdf/2309.09682v1.pdf).

Here you can see some results:
<p align="middle">
<img src="https://user-images.githubusercontent.com/95044968/207450335-1e7f83b2-004f-407d-ba80-9e21468b9e90.gif" width="40%" height="40%"/>
<img src="https://user-images.githubusercontent.com/95044968/207450520-72149d53-a499-4fb5-a8e1-319a00e3c373.gif" width="40%" height="40%"/>
</p>

A full video presentation of the whole project and results is available [here](https://www.youtube.com/watch?v=liIeYU71a5w&t=14s&ab_channel=jiataoding).

If you are interested only in the environment to work on a cleaner and leaner code you can find it here: [environment_only](https://github.com/francescovezzi/quadruped-springs/tree/environment_only).

## Installation

Recommend using conda with python3.7 or higher. After installing conda, this can be done as follows:

`conda create -n [YOUR_ENV_NAME] python=3.7`

To activate the virtualenv: 

`conda activate [YOUR_ENV_NAME]` 

Install all dependencies:

`pip install -r requirements.txt `

## Code structure

- [env](./env) for the quadruped environment files, please see the gym simulation environment [quadruped_gym_env.py](./env/quadruped_gym_env.py), the robot specific functionalities in [quadruped.py](./env/quadruped.py), and config variables in [configs_go1.py](./env/configs_go1.py). You will need to make edits in [quadruped_gym_env.py](./env/quadruped_gym_env.py), and review [quadruped.py](./env/quadruped.py) carefully for accessing robot states and calling functions to solve inverse kinematics, return the leg Jacobian, etc. 
- [go1_description](./go1_description) contains the robot mesh files and urdf. 
- [load_model.py](./load_model.py) provide an interface for loading a RL pre-trained model based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
- [get_demonstrations.py](./get_demonstrations.py) runs a pre-trained model and collects a numpy file containing the trajectory reference performed by the agent during the episode. Such demonstrations are useful for Imitation Learning.
- [hopf_network.py](./hopf_polar.py) provides a CPG class skeleton for various gaits, and maps these to be executed on an instance of the  [quadruped_gym_env](./env/quadruped_gym_env.py) class.

## Training
For training take a look at [rl-baselines3-zoo](https://github.com/francescovezzi/rl-baselines3-zoo/tree/feat/quadruped_support) in the YAML support section.

## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The quadruped environment took inspiration from [Google's motion-imitation repository](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf). 
- Reinforcement learning algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Also see for example [ray[rllib]](https://github.com/ray-project/ray) and [spinningup](https://github.com/openai/spinningup). 

## Acknowledgements

The quadruped env was originally created at [https://gitlab.epfl.ch/bellegar/lr-quadruped-sim](https://gitlab.epfl.ch/bellegar/lr-quadruped-sim) by [Guillaume Bellegarda](https://scholar.google.com/citations?user=YDimn5wAAAAJ&hl=en), who developed the [hopf_network.py](./hopf_polar.py).

<p align="left">
<img src="https://user-images.githubusercontent.com/95044968/207451460-0e4194f5-8522-406f-b071-19176d95d096.gif" width="40%" height="40%"/>
</p>

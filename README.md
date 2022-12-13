# Quadruped-Sim
This repository contains a pybullet based environment for simulating a quadruped robot optionally equipped with parallel elastic actuators (PEA).

<p align="middle">
<img src="https://user-images.githubusercontent.com/95044968/207450335-1e7f83b2-004f-407d-ba80-9e21468b9e90.gif" width="40%" height="40%"/>
<img src="https://user-images.githubusercontent.com/95044968/207450520-72149d53-a499-4fb5-a8e1-319a00e3c373.gif" width="40%" height="40%"/>
</p>

<p align="middle">
<img src="https://user-images.githubusercontent.com/95044968/207451460-0e4194f5-8522-406f-b071-19176d95d096.gif" width="40%" height="40%"/>
</p>


## Installation

Recommend using conda with python3.7 or higher. After installing conda, this can be done as follows:

`conda create -n [YOUR-ENV] python=3.7`

To activate the conda env: 

`conda activate [YOUR-ENV]` 

Your command prompt should now look like: 

`[YOUR-ENV] user@pc:path$`

Install all dependencies:

`pip install -e . `

## Code structure
- [env](./env) for the quadruped environment files, please see the gym simulation environment [quadruped_gym_env.py](./env/quadruped_gym_env.py), the robot specific functionalities in [quadruped.py](./env/quadruped.py). In [control_interface](./env/control_interface) is contained the API for converting action to torques. In [sensors](./env/sensors) quadruped sensors and sensors collections are described. The tasks are defined in [tasks](./env/tasks).
[springs.py](./env/springs.py) contains parallel elastic springs description. User can choose whether to enable them or not.
- [go1_description](./go1_description) contains the robot mesh files, urdf and global configuration variables as joint limits, sensor noise... etc.
- [utils](./utils) for auxiliary files and utilities e.g. plotting and video recording.
- [hopf_network.py](./hopf_polar.py) provides a CPG class skeleton for various gaits, and maps these to be executed on an instance of the  [quadruped_gym_env](./env/quadruped_gym_env.py) class. Please fill in this file carefully. 

## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The quadruped environment took inspiration from [Google's motion-imitation repository](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf). 
- Reinforcement learning algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Also see for example [ray[rllib]](https://github.com/ray-project/ray) and [spinningup](https://github.com/openai/spinningup). 

## Conceptual resources
The CPGs are based on the following papers:
- L. Righetti and A. J. Ijspeert, "Pattern generators with sensory feedback for the control of quadruped locomotion," 2008 IEEE International Conference on Robotics and Automation, 2008, pp. 819-824, doi: 10.1109/ROBOT.2008.4543306. [link](https://ieeexplore.ieee.org/document/4543306)
- M. Ajallooeian, S. Pouya, A. Sproewitz and A. J. Ijspeert, "Central Pattern Generators augmented with virtual model control for quadruped rough terrain locomotion," 2013 IEEE International Conference on Robotics and Automation, 2013, pp. 3321-3328, doi: 10.1109/ICRA.2013.6631040. [link](https://ieeexplore.ieee.org/abstract/document/6631040) 
- M. Ajallooeian, S. Gay, A. Tuleu, A. Spröwitz and A. J. Ijspeert, "Modular control of limit cycle locomotion over unperceived rough terrain," 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2013, pp. 3390-3397, doi: 10.1109/IROS.2013.6696839. [link](https://ieeexplore.ieee.org/abstract/document/6696839) 

## Acknowledgements

- The very first version of this quadruped env was originally created by [Guillaume Bellegarda](https://scholar.google.com/citations?user=YDimn5wAAAAJ&hl=en)
- Thanks to [Antonin Raffin](https://araffin.github.io/) for his support and some code review.

"""This file will be used to test my code changes"""

import os, inspect
# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import time, datetime
import numpy as np

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
random.seed(10)

import quadruped
import configs_a1 as robot_config

_pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
_pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
_pybullet_client.setGravity(0,0,-9.81)
_pybullet_client.loadURDF("plane.urdf")
robot = quadruped.Quadruped(pybullet_client=_pybullet_client,
                            robot_config=robot_config,
                            accurate_motor_model_enabled=True,
                            on_rack=False,
                            enable_springs=False,
                            motor_control_mode="TORQUE")
motor = robot._motor_model

for _ in range(10000):
    torques = np.full(12, 10)
    robot.ApplyAction(torques)
    robot._pybullet_client.stepSimulation()
    time.sleep(0.01)

_pybullet_client.disconnect()

print('************\ndone')
"""This file will be used to test my code changes"""

import inspect
import os

# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import datetime
import random
import time

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

random.seed(10)

import configs_go1 as robot_config
import quadruped

_pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
_pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
_pybullet_client.setGravity(0, 0, -9.81)
_pybullet_client.loadURDF("plane.urdf")
robot = quadruped.Quadruped(
    pybullet_client=_pybullet_client,
    robot_config=robot_config,
    accurate_motor_model_enabled=True,
    on_rack=False,
    enable_springs=True,
    motor_control_mode="TORQUE",
)
motor = robot._motor_model

for _ in range(10000):
    torques = np.full(12, 0)
    robot.ApplyAction(torques)
    robot._pybullet_client.stepSimulation()
    time.sleep(0.01)

_pybullet_client.disconnect()

print("************\ndone")

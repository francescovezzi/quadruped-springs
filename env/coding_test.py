"""This file will be used to test my code changes"""

import os, inspect
# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
os.sys.path.insert(0, currentdir)

import time, datetime
import numpy as np

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
random.seed(10)

import quadruped
import configs_a1 as robot_config


_pybullet_client = bc.BulletClient(connection_mode=p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
plane = p.loadURDF("plane.urdf")
robot = quadruped.Quadruped(pybullet_client=_pybullet_client,
                            robot_config=robot_config,
                            accurate_motor_model_enabled=True)
print(robot._joint_name_to_id)
print()
print(robot._motor_id_list)

for i in range (100):
    time.sleep(1./240.)

p.disconnect()
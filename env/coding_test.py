"""This file will be used to test my code changes"""

import os, inspect
# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
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
import matplotlib.pyplot as plt


_pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
_pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
_pybullet_client.setGravity(0,0,0)
plane = _pybullet_client.loadURDF("plane.urdf")
robot = quadruped.Quadruped(pybullet_client=_pybullet_client,
                            robot_config=robot_config,
                            accurate_motor_model_enabled=True,
                            on_rack=True)
num_motors = robot_config.NUM_MOTORS

print(robot._motor_id_list)

time_sleep = 1./2400.
timesteps = 2000
time_sequence = [time_sleep * i for i in range(timesteps)]
fr_leg_history = np.zeros((3,timesteps))
fr_leg_history_tau = np.zeros((3,timesteps))
desired_motor_angles = np.full(12, 0)

for i in range (timesteps):
    desired_motor_angles = [0,np.pi/4, -np.pi/2]*4
    
    # fr_leg_tau = robot.ApplyAction(desired_motor_angles)[0:3]
    for num, mot_id in enumerate(robot._motor_id_list):
        robot._SetDesiredMotorAngleById(mot_id, 0)

    robot._SetDesiredMotorAngleByName('FR_calf_joint', -1)

    fr_leg_angles = robot.GetMotorAngles()[0:3]
    fr_leg_tau = robot.GetMotorTorques()[0:3]
    fr_leg_history[:,i] = fr_leg_angles
    fr_leg_history_tau[:,i] = fr_leg_tau

    
    p.stepSimulation()
    time.sleep(time_sleep)

p.disconnect()


##########################
# Plot Stuff #
##########################
fig_fr_angles, ax_fr_angles = plt.subplots()
ax_fr_angles.plot(time_sequence, np.transpose(fr_leg_history))
ax_fr_angles.set_title('FR_LEG_STATE(t)')
ax_fr_angles.legend(('hip', 'thigh', 'calf'))
plt.show()

fig_fr_tau, ax_fr_tau = plt.subplots()
ax_fr_tau.plot(time_sequence, np.transpose(fr_leg_history_tau))
ax_fr_tau.set_title('FR_LEG_TAU(t)')
ax_fr_tau.legend(('hip', 'thigh', 'calf'))
plt.show()
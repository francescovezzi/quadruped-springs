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


def restart_simulation(_pybullet_client, robot, height=0):
    _pybullet_client.resetSimulation()
    _pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    _pybullet_client.setGravity(0,0,-9.81)
    _pybullet_client.loadURDF("plane.urdf")
    robot.Reset_height(h=height)


_pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
robot = quadruped.Quadruped(pybullet_client=_pybullet_client,
                            robot_config=robot_config,
                            accurate_motor_model_enabled=True,
                            on_rack=True)


restart_simulation(_pybullet_client, robot=robot)


time_sleep = 1./2400.
timesteps = 20000
time_sequence = [time_sleep * i for i in range(timesteps)]
fr_leg_history = np.zeros((3,timesteps))
fr_leg_history_tau = np.zeros((3,timesteps))
fr_leg_history_tau_spring = np.zeros((3,timesteps))

for i in range (timesteps):
    
    robot.ApplySpringAction()

    fr_leg_angles = robot.GetMotorAngles()[0:3]
    fr_leg_tau = robot._applied_motor_torque[0:3]
    fr_leg_tau_spring = robot._spring_torque[0:3]
    
    fr_leg_history[:,i] = fr_leg_angles
    fr_leg_history_tau[:,i] = fr_leg_tau
    fr_leg_history_tau_spring[:,i] = fr_leg_tau_spring
        
    _pybullet_client.stepSimulation()
    time.sleep(time_sleep)

_pybullet_client.disconnect()


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

fig_fr_tau_spring, ax_fr_tau_spring = plt.subplots()
ax_fr_tau_spring.plot(time_sequence, np.transpose(fr_leg_history_tau_spring))
ax_fr_tau_spring.set_title('FR_LEG_TAU_SPRING(t)')
ax_fr_tau_spring.legend(('hip', 'thigh', 'calf'))
plt.show()
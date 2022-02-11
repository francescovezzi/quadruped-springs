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

def change_trunk_mass(_pybullet_client, robot, beta=0):
    """ Modify the trunk mass as new_mass = (1 + beta) * old_mass.
    Obs:
    -It will automatically change the inertia
    -it will not modify the URDF files so in the robot class
        the mass is still setted to the old value

    Args:
        _pybullet_client (bullet_client): bullet_client
        robot (class Quadruped): The class
        beta (int, optional): parameter to change mass value. Defaults to 0.
    """
    trunkId = robot._chassis_link_ids[0]
    old_mass = _pybullet_client.getDynamicsInfo(robot.quadruped,
                                                trunkId)[0]
    new_mass = old_mass * (1 + beta)
    _pybullet_client.changeDynamics(robot.quadruped,
                                    trunkId,
                                    mass=new_mass)


_pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
robot = quadruped.Quadruped(pybullet_client=_pybullet_client,
                            robot_config=robot_config,
                            accurate_motor_model_enabled=True,
                            on_rack=False)
motor = robot._motor_model

time_sleep = 1./2400.
timesteps = 20000
time_sequence = [time_sleep * i for i in range(timesteps)]
fr_leg_history = np.zeros((3,timesteps))
fr_leg_history_tau = np.zeros((3,timesteps))
fr_leg_history_tau_spring = np.zeros((3,timesteps))

###############################
# Define parameters for test
###############################

list_stiffness = [ [15, 8.6, 5.1]]
list_rest_angles = [ [0, np.pi/4, -0.7]]

restart_simulation(_pybullet_client, robot=robot, height=0.2)
motor._setSpringStiffness(list_stiffness[0])
motor._setSpringRestAngle(list_rest_angles[0])
change_trunk_mass(_pybullet_client, robot, beta=0.5)

for i in range (timesteps):
    
    # robot.ApplySpringAction()

    # Log values
    fr_leg_angles = robot.GetMotorAngles()[0:3]
    fr_leg_tau = robot._applied_motor_torque[0:3]
    fr_leg_tau_spring = robot._spring_torque[0:3]
    
    fr_leg_history[:,i] = fr_leg_angles
    fr_leg_history_tau[:,i] = fr_leg_tau
    fr_leg_history_tau_spring[:,i] = fr_leg_tau_spring

    # simulate    
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
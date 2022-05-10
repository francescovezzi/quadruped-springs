"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
from sys import platform

import matplotlib
import numpy as np

if platform == "darwin":  # mac
    import PyQt5

    matplotlib.use("Qt5Agg")
else:  # linux
    matplotlib.use("TkAgg")

from matplotlib import pyplot as plt

from quadruped_spring.env.quadruped_gym_env import QuadrupedGymEnv


class HopfNetwork:
    """CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

    Foot Order is FR, FL, RR, RL
    (Front Right, Front Left, Rear Right, Rear Left)
    """

    def __init__(
        self,
        mu=2,  # 1**2,            # converge to sqrt(mu)
        omega_swing=1 * 2 * np.pi,  # MUST EDIT
        omega_stance=1 * 2 * np.pi,  # MUST EDIT
        gait="TROT",  # change depending on desired gait
        coupling_strength=1,  # coefficient to multiply coupling matrix
        couple=True,  # should couple
        time_step=0.001,  # time step
        ground_clearance=0.05,  # foot swing height
        ground_penetration=0.01,  # foot stance penetration into ground
        robot_height=0.25,  # in nominal case (standing)
        des_step_len=0.04,  # desired step length
    ):

        ###############
        # initialize CPG data structures: amplitude is row 0, and phase is row 1
        self.X = np.zeros((2, 4))

        # save parameters
        self._mu = mu
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)

        # set oscillator initial conditions
        self.X[0, :] = np.random.rand(4) * 0.1
        self.X[1, :] = self.PHI[0, :]

        # save body and foot shaping
        self._ground_clearance = ground_clearance
        self._ground_penetration = ground_penetration
        self._robot_height = robot_height
        self._des_step_len = des_step_len

        self.X_list = []
        self.dX_list = []

    def _set_gait(self, gait):
        """For coupling oscillators in phase space.
        [TODO] update all coupling matrices
        """
        self.PHI_trot = np.zeros((4, 4))
        self.PHI_trot[0] = np.array([0, -np.pi, -np.pi, 0])
        self.PHI_trot[1] = np.array([np.pi, 0, 0, np.pi])
        self.PHI_trot[2] = np.array([np.pi, 0, 0, np.pi])
        self.PHI_trot[3] = np.array([0, -np.pi, -np.pi, 0])

        self.PHI_walk = np.zeros((4, 4))
        self.PHI_walk[0] = np.array([0, -np.pi, -np.pi / 2.0, np.pi / 2.0])
        self.PHI_walk[1] = np.array([np.pi, 0, np.pi / 2.0, 3.0 * np.pi / 2.0])
        self.PHI_walk[2] = np.array([np.pi / 2.0, -np.pi / 2.0, 0, np.pi])
        self.PHI_walk[3] = np.array([-np.pi / 2.0, -3.0 * np.pi / 2.0, -np.pi, 0])

        self.PHI_bound = np.zeros((4, 4))
        self.PHI_bound[0] = np.array([0, 0, -np.pi, -np.pi])
        self.PHI_bound[1] = np.array([0, 0, -np.pi, -np.pi])
        self.PHI_bound[2] = np.array([np.pi, np.pi, 0, 0])
        self.PHI_bound[3] = np.array([np.pi, np.pi, 0, 0])

        self.PHI_pace = np.zeros((4, 4))
        self.PHI_pace[0] = np.array([0, -np.pi, 0, -np.pi])
        self.PHI_pace[1] = np.array([np.pi, 0, np.pi, 0])
        self.PHI_pace[2] = np.array([0, -np.pi, 0, -np.pi])
        self.PHI_pace[3] = np.array([np.pi, 0, np.pi, 0])

        if gait == "TROT":
            print("TROT")
            self.PHI = self.PHI_trot
        elif gait == "PACE":
            print("PACE")
            self.PHI = self.PHI_pace
        elif gait == "BOUND":
            print("BOUND")
            self.PHI = self.PHI_bound
        elif gait == "WALK":
            print("WALK")
            self.PHI = self.PHI_walk
        else:
            raise ValueError(gait + "not implemented.")

    def update(self):
        """Update oscillator states."""

        # update parameters, integrate
        self._integrate_hopf_equations()

        # map CPG variables to Cartesian foot xz positions (Equations 8, 9)
        x = np.zeros(4)
        z = np.zeros(4)
        for i in range(4):
            r = self.X[0, i]
            theta = self.X[1, i]
            x[i] = -self._des_step_len * r * np.cos(theta)
            if np.sin(theta) > 0:
                z[i] = -self._robot_height + self._ground_clearance * np.sin(theta)
            else:
                z[i] = -self._robot_height + self._ground_penetration * np.sin(theta)

        return x, z

    def _integrate_hopf_equations(self):
        """Hopf polar equations and integration. Use equations 6 and 7."""
        # bookkeeping - save copies of current CPG states
        X = self.X.copy()
        X_dot = np.zeros((2, 4))
        alpha = 50

        # loop through each leg's oscillator
        for i in range(4):
            # get r_i, theta_i from X
            r = X[0, i]
            theta = X[1, i]
            # compute r_dot (Equation 6)
            r_dot = alpha * (self._mu - r**2) * r
            # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
            theta_dot = 0.0
            if np.sin(theta) > 0:
                theta_dot = self._omega_swing
            else:
                theta_dot = self._omega_stance

            # loop through other oscillators to add coupling (Equation 7)
            if self._couple:
                for j in range(4):
                    if j != i:
                        theta_dot += X[0, j] * self._coupling_strength * np.sin(X[1, j] - theta - self.PHI[i, j])

            # set X_dot[:,i]
            X_dot[:, i] = [r_dot, theta_dot]

        # integrate
        self.X += self._dt * X_dot
        # mod phase variables to keep between 0 and 2pi
        self.X[1, :] = self.X[1, :] % (2 * np.pi)

        self.X_list.append(self.X.copy())
        self.dX_list.append(X_dot)


if __name__ == "__main__":

    ADD_CARTESIAN_PD = True
    TIME_STEP = 0.001
    foot_y = 0.0838  # this is the hip length
    sideSign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)

    env = QuadrupedGymEnv(
        render=True,  # visualize
        on_rack=False,  # useful for debugging!
        isRLGymInterface=False,  # not using RL
        time_step=TIME_STEP,
        action_repeat=1,
        motor_control_mode="TORQUE",
        add_noise=False,  # start in ideal conditions
        record_video=False,
        enable_springs=False,
        enable_env_randomization=False,
    )

    # initialize Hopf Network, supply gait
    # TROT
    omega_swing = 16.0 * np.pi
    omega_stance = 4.0 * np.pi
    # WALK
    # omega_swing = 24.0 * np.pi
    # omega_stance = 25.0 * np.pi
    # PACE
    # omega_swing = 20.0 * np.pi
    # omega_stance = 20.0 * np.pi
    # BOUND
    # omega_swing = 10.0 * np.pi
    # omega_stance = 40.0 * np.pi

    d_swing = np.pi / omega_swing
    print("Swing duration: ", d_swing)
    d_stance = np.pi / omega_stance
    print("Stance duration: ", d_stance)
    d_stride = d_swing + d_stance
    print("Stride duration: ", d_stride)
    duty_factor = d_stance / d_stride
    print("Duty Factor: ", duty_factor)

    cpg = HopfNetwork(gait="TROT", omega_swing=omega_swing, omega_stance=omega_stance, time_step=TIME_STEP)

    T = 10.0
    TEST_STEPS = int(T / (TIME_STEP))
    t = np.arange(TEST_STEPS) * TIME_STEP

    # initialize data structures to save CPG and robot states
    xs_list = np.zeros((TEST_STEPS, 4))
    zs_list = np.zeros((TEST_STEPS, 4))
    x_list = np.zeros((TEST_STEPS, 4))
    z_list = np.zeros((TEST_STEPS, 4))

    q_list = np.zeros((TEST_STEPS, 3))
    qd_list = np.zeros((TEST_STEPS, 3))

    v_list = np.zeros(TEST_STEPS)
    P_list = np.zeros(TEST_STEPS)

    ############## Sample Gains
    # joint PD gains
    kp = np.array([150, 70, 70])
    kd = np.array([2, 0.5, 0.5])
    # Cartesian PD gains
    kpCartesian = np.diag([2500] * 3)
    kdCartesian = np.diag([40] * 3)

    for j in range(TEST_STEPS):
        # initialize torque array to send to motors
        action = np.zeros(12)
        # get desired foot positions from CPG
        xs, zs = cpg.update()
        # get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
        q = env.robot.GetMotorAngles()
        dq = env.robot.GetMotorVelocities()

        v = np.array(env.robot.GetBaseLinearVelocity())[:2]
        v_list[j] = np.linalg.norm(v)

        tau_ = env.robot.GetMotorTorques()
        P_list[j] = np.dot(dq, tau_)

        # loop through desired foot positions and calculate torques
        for i in range(4):
            q_i = q[3 * i : 3 * (i + 1)]
            dq_i = dq[3 * i : 3 * (i + 1)]
            # initialize torques for legi
            tau = np.zeros(3)
            # get desired foot i pos (xi, yi, zi) in leg frame
            xyz_d = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
            # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
            leg_q = env.robot.ComputeInverseKinematics(i, xyz_d)
            # Add joint PD contribution to tau for leg i (Equation 4)
            tau += -kp * (q_i - leg_q) - kd * dq_i

            if i == 0:
                q_list[j] = q_i
                qd_list[j] = leg_q

            # add Cartesian PD contribution
            if ADD_CARTESIAN_PD:
                # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
                J, xyz = env.robot.ComputeJacobianAndPosition(i)
                x_list[j, i] = xyz[0]
                z_list[j, i] = xyz[2]
                # Get current foot velocity in leg frame (Equation 2)
                dxyz = J @ dq_i
                F_foot = -kpCartesian @ (xyz - xyz_d) - kdCartesian @ dxyz
                # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
                tau += J.T @ F_foot

            # Set tau for legi in action vector
            action[3 * i : 3 * i + 3] = tau

        # send torques to robot and simulate TIME_STEP seconds
        env.step(action)

        xs_list[j] = xs
        zs_list[j] = zs

    # fig = plt.figure(1, figsize=(16, 6), dpi=200, facecolor="w", edgecolor="k")
    # k = 1000
    # plt.plot(xs_list[-k:, 0], zs_list[-k:, 0], "b", label=r"x vs z (des)")
    # plt.plot(x_list[-k:, 0], z_list[-k:, 0], "b--", label=r"x vs z")
    # plt.legend(loc="right")
    # plt.xlabel(r"$x vs z$")
    # plt.show()

    exit()
    #####################################################
    # PLOTS
    #####################################################
    fig = plt.figure(1, figsize=(16, 6), dpi=200, facecolor="w", edgecolor="k")
    k = 1000
    plt.plot(t[:k], x_list[:k, 0], "b", label=r"$x$")
    plt.plot(t[:k], xs_list[:k, 0], "b--", label=r"$x_{des}$")
    plt.plot(t[:k], z_list[:k, 0], "r", label=r"$z$")
    plt.plot(t[:k], zs_list[:k, 0], "r--", label=r"$z_{des}$")
    plt.legend(loc="right")
    plt.xlabel(r"$t$")

    fig = plt.figure(2, figsize=(16, 6), dpi=200, facecolor="w", edgecolor="k")
    k = 1000
    plt.plot(t[:k], q_list[:k, 0], "b", label=r"$q_0$")
    plt.plot(t[:k], qd_list[:k, 0], "b--", label=r"$q_{0,des}$")
    plt.plot(t[:k], q_list[:k, 1], "r", label=r"$q_1$")
    plt.plot(t[:k], qd_list[:k, 1], "r--", label=r"$q_{1,des}$")
    plt.plot(t[:k], q_list[:k, 2], "g", label=r"$q_2$")
    plt.plot(t[:k], qd_list[:k, 2], "g--", label=r"$q_{2,des}$")
    plt.legend(loc="right")
    plt.xlabel(r"$t$")

    plt.figure(3)
    v_list_ = np.zeros_like(v_list)
    v_list_[0] = v_list[0]
    for i in range(len(v_list) - 1):
        v_list_[i + 1] = 0.999 * v_list_[i] + 0.001 * v_list[i + 1]
    plt.plot(t, v_list_)

    plt.figure(4)
    P_list_ = np.zeros_like(P_list)
    P_list_[0] = P_list[0]
    for i in range(len(P_list) - 1):
        P_list_[i + 1] = 0.999 * P_list_[i] + 0.001 * P_list[i + 1]
    # plt.plot(t, P_list)
    plt.plot(t, P_list_)

    # fig = plt.figure(1)
    # for i in range(4):
    #  plt.plot(t, xs_list[:,i], label='x_'+str(i))
    # plt.legend()
    # plt.show()

    # fig = plt.figure(2)
    # for i in range(4):
    #  plt.plot(t, zs_list[:,i], label='x_'+str(i))
    # plt.legend()
    # plt.show()

    # CPG States
    # foot_names = []
    # foot_names.append('FR')
    # foot_names.append('FL')
    # foot_names.append('RR')
    # foot_names.append('RL')

    # r_list = np.zeros((len(cpg.X_list), 4))
    # theta_list = np.zeros_like((r_list))
    # dr_list = np.zeros_like((r_list))
    # dtheta_list = np.zeros_like((r_list))
    # for i in range(len(cpg.X_list)):
    #  r_list[i] = cpg.X_list[i][0,:]
    #  theta_list[i] = cpg.X_list[i][1,:]
    #  dr_list[i] = cpg.dX_list[i][0,:]
    #  dtheta_list[i] = cpg.dX_list[i][1,:]

    # fig = plt.figure(3, figsize=(16, 6), dpi=200, facecolor='w', edgecolor='k')
    # k = 1000
    # for i in range(4):
    #  plt.subplot(411+i)
    #  plt.plot(t[:k], r_list[:k,i] / np.max(r_list[:k,i]), label=r'$r$')
    #  plt.plot(t[:k], theta_list[:k,i] / np.max(theta_list[:k,i]), label=r'$\theta$')
    #  plt.plot(t[:k], dr_list[:k,i] / np.max(dr_list[:k,i]), label=r'$\dot{r}$')
    #  plt.plot(t[:k], dtheta_list[:k,i] / np.max(dtheta_list[:k,i]), label=r'$\dot{\theta}$')
    #  if i == 0:
    #    plt.legend(loc='upper right', fontsize=10)
    #  if i != 3:
    #    plt.xticks([])
    #  else:
    #    plt.xlabel(r'$t$')
    #  plt.yticks([0,1])
    #  plt.ylabel(foot_names[i])

    # plt.tight_layout()
    # plt.show()

    # example
    # fig = plt.figure()
    # plt.plot(t,joint_pos[1,:], label='FR thigh')
    # plt.legend()
    # plt.show()

    plt.show()

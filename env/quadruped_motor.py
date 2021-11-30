"""Motor model for quadrupeds."""

import collections
import numpy as np

NUM_MOTORS = 12

CONTROL_MODES = [ "TORQUE","PD" ]

class QuadrupedMotorModel(object):
  """A simple motor model for A1.

    When in POSITION mode, the torque is calculated according to the difference
    between current and desired joint angle, as well as the joint velocity.
    For more information about PD control, please refer to:
    https://en.wikipedia.org/wiki/PID_controller.

    The model supports a HYBRID mode in which each motor command can be a tuple
    (desired_motor_angle, position_gain, desired_motor_velocity, velocity_gain,
    torque).

  """

  def __init__(self,
               kp=60,
               kd=1,
               torque_limits=None,
               motor_control_mode="PD"):
    self._kp = kp
    self._kd = kd
    self._torque_limits = torque_limits
    if torque_limits is not None:
      if isinstance(torque_limits, (collections.Sequence, np.ndarray)):
        self._torque_limits = np.asarray(torque_limits)
      else:
        self._torque_limits = np.full(NUM_MOTORS, torque_limits)
    self._motor_control_mode = motor_control_mode
    self._strength_ratios = np.full(NUM_MOTORS, 1)

  def convert_to_torque(self,
                        motor_commands,
                        motor_angle,
                        motor_velocity,
                        motor_control_mode=None):
    """Convert the commands (position control or torque control) to torque.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      motor_angle: The motor angle observed at the current time step. It is
        actually the true motor angle observed a few milliseconds ago (pd
        latency).
      motor_velocity: The motor velocity observed at the current time step, it
        is actually the true motor velocity a few milliseconds ago (pd latency).
      motor_control_mode: A MotorControlMode enum.

    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    if not motor_control_mode:
      motor_control_mode = self._motor_control_mode

    # No processing for motor torques
    # Edit: SHOULD still clip torque values
    if motor_control_mode is "TORQUE":
      assert len(motor_commands) == NUM_MOTORS
      motor_torques = self._strength_ratios * motor_commands
      motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                              self._torque_limits)
      #print('actual motor', motor_torques)
      return motor_torques, motor_torques

    desired_motor_angles = None
    desired_motor_velocities = None
    kp = None
    kd = None
    additional_torques = np.full(NUM_MOTORS, 0)
    if motor_control_mode is "PD":
      assert len(motor_commands) == NUM_MOTORS
      kp = self._kp
      kd = self._kd
      desired_motor_angles = motor_commands
      desired_motor_velocities = np.full(NUM_MOTORS, 0)
    else:
      raise ValueError("Motor model should only be torque or position control.")

    motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (
        motor_velocity - desired_motor_velocities) + additional_torques
    motor_torques = self._strength_ratios * motor_torques
    if self._torque_limits is not None:
      if len(self._torque_limits) != len(motor_torques):
        raise ValueError(
            "Torque limits dimension does not match the number of motors.")
      motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                              self._torque_limits)

    return motor_torques, motor_torques

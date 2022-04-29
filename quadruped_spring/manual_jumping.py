import inspect
import os
import time

import gym
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)

import argparse

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.evaluate_metric import EvaluateMetricJumpOnPlace
from utils.timer import Timer

# from utils.monitor_state import MonitorState


class JumpingStateMachine(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._settling_duration_time = 1  # seconds
        self._couching_duration_time = 4  # seconds
        self._settling_duration_steps = self._settling_duration_time / (self.env._time_step * self.env._action_repeat)
        self._couching_duration_steps = self._couching_duration_time / (self.env._time_step * self.env._action_repeat)
        assert self._couching_duration_steps >= 1000, "couching duration steps number should be >= 1000"
        self._states = {"settling": 0, "couching": 1, "jumping_ground": 2, "jumping_air": 3, "landing": 4}
        self._state = self._states["settling"]
        self._flying_up_counter = 0
        self._actions = {
            0: self.settling_action,
            1: self.couching_action,
            2: self.jumping_explosive_action,
            3: self.jumping_flying_action,
        }
        self._total_sim_steps = 9000
        self.max_height = 0.0
        self._step_counter = 0

        self._time_step = self.env._time_step
        self._robot_config = self.env._robot_config
        self._enable_springs = self.env._enable_springs
        self._jump_end = False
        self._flight_timer = Timer()

        self._default_action = self.env._compute_action_from_command(
            self._robot_config.INIT_MOTOR_ANGLES,
            self._robot_config.RL_LOWER_ANGLE_JOINT,
            self._robot_config.RL_UPPER_ANGLE_JOINT,
        )

    def temporary_switch_motor_control_gain(foo):
        def wrapper(self, *args, **kwargs):
            """Settle robot and add noise to init configuration."""
            if self.env._enable_springs:
                ret = foo(self, *args, **kwargs)
            else:
                tmp_save_motor_kp = self.env.robot._motor_model._kp
                tmp_save_motor_kd = self.env.robot._motor_model._kd
                self.env.robot._motor_model._kp = 60
                self.env.robot._motor_model._kd = 3.0
                ret = foo(self, *args, **kwargs)
                self.env.robot._motor_model._kp = tmp_save_motor_kp
                self.env.robot._motor_model._kd = tmp_save_motor_kd
            return ret

        return wrapper

    def compute_action(self):
        if self._state == self._states["landing"]:
            raise ValueError("this phase should be managed by")
        return self._actions[self._state]()

    def update_state(self):
        if self._step_counter <= self._settling_duration_steps:
            actual_state = self._states["settling"]
        elif self._step_counter <= self._settling_duration_steps + self._couching_duration_steps:
            actual_state = self._states["couching"]
        else:
            if not self._jump_end:
                if not self.is_flying():
                    actual_state = self._states["jumping_ground"]
                else:
                    if not self.flight_time_gone():
                        actual_state = self._states["jumping_air"]
                    else:
                        self._jump_end = True
            if self._jump_end:
                actual_state = self._states["landing"]

        self.max_height = max(self.max_height, self.env.robot.GetBasePosition()[2])
        # if self._state != actual_state:
        #     print('********************')
        #     print(f'{self._state} -> {actual_state}')
        #     print(f'joint config is: {self.env.robot.GetMotorAngles()}')
        #     print(f'sim time is: {self.env.get_sim_time()}')
        #     print('********************')
        self._state = actual_state

    def flight_time_gone(self):
        if not self._flight_timer.already_started():
            _, _, self.vz = self.env.robot.GetBaseLinearVelocity()
            flight_time = self.vz / 9.81
            actual_time = self.env.get_sim_time()
            self._flight_timer.start_timer(timer_time=actual_time, start_time=actual_time, delta_time=flight_time)
        self._flight_timer.update_time(self.env.get_sim_time())
        return self._flight_timer.time_up()

    def base_velocity(self):
        q_dot = self.env.robot.GetMotorVelocities()
        _, _, _, feet_contact = self.env.robot.GetContactInfo()
        base_vel = np.zeros(3)
        count = 1
        for in_contact, id in zip(feet_contact, list(range(4))):
            if in_contact:
                Jac, _ = self.env.robot.ComputeJacobianAndPosition(id)
                vel = -Jac @ q_dot[3 * id : 3 * (id + 1)]
                base_vel = base_vel + (vel - base_vel) / count
                count += 1
        return base_vel

    def settling_action(self):
        return self._default_action

    def couching_action(self):
        max_action_calf = -1
        min_action_calf = self._default_action[2]
        max_action_thigh = 0.9
        min_action_thigh = self._default_action[1]
        i = self._step_counter
        i_min = self._settling_duration_steps
        i_max = i_min + self._couching_duration_steps - 500
        action_thigh = self.generate_ramp(i, i_min, i_max, min_action_thigh, max_action_thigh)
        action_calf = self.generate_ramp(i, i_min, i_max, min_action_calf, max_action_calf)
        torques = np.array([0, action_thigh, action_calf] * 4)
        # torques = np.array([0, 0.5, 1] * 4)
        return torques

    def jumping_explosive_action(self):
        if self.env._enable_springs:
            coeff = -0.1
        else:
            coeff = 0.0
        action_front = np.array([0, 0, coeff * 1] * 2)
        action_rear = np.array([0, 0, 1] * 2)
        jump_action = np.concatenate((action_front, action_rear))
        return jump_action

    def jumping_flying_action(self):
        action = np.zeros(12)
        action = np.array([0, 0, 1] * 4)
        return action

    def landing_step(self):
        self.robot.ApplyAction(self.jumping_landing_torque())
        if self.env._is_render:
            self.env._render_step_helper()
        self.env._pybullet_client.stepSimulation()

    def robot_stopped(self):
        vel = self.env.robot.GetBaseLinearVelocity()
        vel_module = np.sqrt(np.dot(vel, vel))
        return vel_module < 0.01

    def stop_landing(self):
        return self.env.terminated or self.robot_stopped()

    @temporary_switch_motor_control_gain
    def landing_phase(self):
        command = self.env._robot_config.INIT_MOTOR_ANGLES
        action = self.env._compute_action_from_command(
            command, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT
        )
        done = False
        while not done:
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def is_flying(self):
        _, _, _, feetInContactBool = self.env.robot.GetContactInfo()
        return np.all(1 - np.array(feetInContactBool))
    
    @staticmethod
    def generate_ramp(i, i_min, i_max, u_min, u_max) -> float:
        if i < i_min:
            return u_min
        elif i > i_max:
            return u_max
        else:
            return u_min + (u_max - u_min) * (i - i_min) / (i_max - i_min)

    def step(self, action):

        obs, reward, done, infos = self.env.step(action)
        self._step_counter += 1

        self.update_state()

        if self._state == self._states["landing"]:
            _, reward, done, infos = self.landing_phase()

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()


def build_env(enable_springs=False):
    env_config = {}
    env_config["enable_springs"] = enable_springs
    env_config["enable_springs"] = True
    env_config["render"] = True
    env_config["on_rack"] = False
    env_config["enable_joint_velocity_estimate"] = False
    env_config["isRLGymInterface"] = True
    env_config["robot_model"] = "GO1"
    env_config["motor_control_mode"] = "PD"
    env_config["action_repeat"] = 1
    env_config["record_video"] = False
    env_config["action_space_mode"] = "DEFAULT"
    env_config["task_env"] = "JUMPING_ON_PLACE_ABS_HEIGHT_TASK"

    if fill_line:
        env_config["render"] = True
    env = QuadrupedGymEnv(**env_config)
    return env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-springs", action="store_true", default=False, help="enable springs")
    parser.add_argument("--fill-line", action="store_true", default=False, help="fill line in report.txt")
    args = parser.parse_args()
    enable_springs = args.enable_springs
    fill_line = args.fill_line

    env = build_env(enable_springs=enable_springs)
    env = JumpingStateMachine(env)
    sim_steps = env._total_sim_steps + 3000
    # env = MonitorState(env=env, path="logs/plots/manual_jumping_with_springs", rec_length=sim_steps)
    env = EvaluateMetricJumpOnPlace(env)
    done = False
    while not done:
        action = env.compute_action()
        # action = np.zeros(12)
        obs, reward, done, info = env.step(action)
        # print(env.robot.GetMotorVelocities()-env.get_joint_velocity_estimation())
    # env.release_plots()
    print("******")
    print(f"reward -> {reward}")
    print(f"min_height -> {env.get_metric().height_min}")
    print("******")
    if fill_line:
        report_path = os.path.join(current_dir, "logs", "models", "performance_report.txt")
        with open(report_path, "w") as f:
            f.write(env.print_first_line_table())
            if enable_springs:
                f.write(env.fill_line(id="ManualWithSprings"))
            else:
                f.write(env.fill_line(id="ManualWithoutSprings"))
    else:
        env.print_metric()
    env.close()
    print("end")

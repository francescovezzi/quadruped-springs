import gym

import numpy as np

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.monitor_state2 import MonitorState


class JumpingStateMachine(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        if self.env._isRLGymInterface:
            raise ValueError('disable RLGymInterface in env_configs')
        if self.env._motor_control_mode != "TORQUE":
            raise ValueError('motor control mode should be TORQUE')
        self._settling_duration_steps = 1000
        self._couching_duration_steps = 4000
        assert self._couching_duration_steps >= 1000, 'couching duration steps number should be >= 1000'
        self._states = {'settling': 0, 'couching': 1, 'jumping_ground': 2, 'jumping_air': 3, 'landing': 4}
        self._state = self._states['settling']
        self._flying_up_counter = 0
        self._actions = {0: self._settling_action, 1: self._couching_action, 2: self._jumping_explosive_action, 3: self._jumping_flying_action, 4: self._jumping_landing_action}
        self._total_sim_steps = 9000
        self._max_height = 0.0
        self._step_counter = 0


    def _compute_action(self):
        return self._actions[self._state]()
    
    def _compensate_spring(self):
        spring_action = self.env.robot._spring_torque
        return -np.array(spring_action)

    def _update_state(self):
        if self._step_counter <= self._settling_duration_steps:
            actual_state = self._states['settling']
        elif self._step_counter <= self._settling_duration_steps + self._couching_duration_steps:
            actual_state = self._states['couching']
            # print(self.env.robot.GetBasePosition()[2])
        else:
            if self._all_feet_in_contact():
                actual_state = self._states['jumping_ground']
            else:
                if self._is_landing():
                    actual_state = self._states['landing']
                else:
                    actual_state = self._states['jumping_air']
        self._max_height = max(self._max_height, self.env.robot.GetBasePosition()[2])
        self._state = actual_state
        
    def _settling_action(self):
        if self.env._enable_springs:
            config_des = np.array(self.env._robot_config.SPRINGS_REST_ANGLE * 4)
            config_init = self.env._robot_config.INIT_MOTOR_ANGLES
            config_ref = self.generate_ramp(self._step_counter, 0, self._settling_duration_steps -200, config_init, config_des)
            action = self.angle_ref_to_command(config_ref)
        else:
            config_des = self.env._robot_config.INIT_MOTOR_ANGLES
            action = self.angle_ref_to_command(config_des)
        return action
        
    # def _couching_action(self):
    #     i_min = self._settling_duration_steps
    #     i_max = self._couching_duration_steps - 150 + i_min
    #     config_init = self.env._robot_config.INIT_MOTOR_ANGLES
    #     config_des = self.height_to_theta_des(0.14)
    #     config_ref = [self.generate_ramp(self._step_counter, i_min, i_max, config_init[j], config_des[j]) for j in range(12)]
    #     command = self.angle_ref_to_command(config_ref)
    #     return command

    def _couching_action(self):
        max_torque = 35.55*0.9
        min_torque = 0
        i = self._step_counter
        i_min = self._settling_duration_steps
        i_max = i_min + self._couching_duration_steps - 500
        torque_thigh = self.generate_ramp(i, i_min, i_max, 0, 20)
        torque_calf = self.generate_ramp(i, i_min, i_max, min_torque, max_torque)
        torques = np.array([0,torque_thigh,-torque_calf]*4)
        return torques
    
    def _jumping_explosive_action(self):
        coeff = 1.5
        f_rear = 150
        f_front = coeff * f_rear
        jump_command = np.full(12, 0)
        for i in range(4):
            if i < 2:
                f = f_front
            else:
                f = f_rear
            jump_command[3 * i : 3 * (i + 1)] = self.map_force_to_tau([0, 0, -f], i)
            jump_command[3*i] = 0
        print(jump_command)
        return jump_command
    
    def _jumping_flying_action(self):
        action = np.full(12,0)
        return action
    
    def _jumping_landing_action(self):
        config_des = np.array(self.env._robot_config.SPRINGS_REST_ANGLE * 4)
        config_des = np.array(self.env._robot_config.INIT_MOTOR_ANGLES)
        q = self.robot.GetMotorAngles()
        dq = self.robot.GetMotorVelocities()
        if self.env._enable_springs:
            kp = 70
            kd = 1.0
        else:
            kp = 55
            kd = 0.8
        torque = -kp * (q - config_des) - kd * dq
        config_des = self.env._robot_config.INIT_MOTOR_ANGLES
        # action = self.angle_ref_to_command(config_des)
        action = torque
        return action
        
    def _all_feet_in_contact(self):
        _, _, _, feetInContactBool = self.env.robot.GetContactInfo()
        return np.all(feetInContactBool)
    
    def _is_landing(self):
        if self._flying_up_counter >= 20:
            return True
        else:
            self._flying_up_counter += 1
            return False
        
    def generate_ramp(self, i, i_min, i_max, u_min, u_max) -> float:
        if i < i_min:
            return u_min
        elif i > i_max:
            return u_max
        else:
            return u_min + (u_max - u_min) * (i - i_min) / (i_max - i_min)

    def angle_ref_to_command(self, angles_ref):
        q = self.robot.GetMotorAngles()
        dq = self.robot.GetMotorVelocities()
        if self.env._enable_springs:
            kp = 70
            kd = 0.8
        else:
            kp = 55
            kd = 0.8
        torque = -kp * (q - angles_ref) - kd * dq
        # print(q-angles_ref)
        return torque
    
    def height_to_theta_des(self, h):
        l = self.env._robot_config.THIGH_LINK_LENGTH
        theta_thigh = np.arccos(h / (2 * l))
        theta_des = np.array([0, theta_thigh, -2 * theta_thigh] * 4)
        return theta_des
    
    def map_force_to_tau(self, F_foot, i):
        J, _ = self.env.robot.ComputeJacobianAndPosition(i)
        tau = J.T @ F_foot
        return tau

        
    def step(self, action):

        obs, reward, done, infos = self.env.step(action)
        self._step_counter += 1
        
        self._update_state()

        return obs, reward, done, infos
    
    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def close(self):
        self.env.close()
    
def build_env():
    env_config = {}
    env_config["enable_springs"] = True
    env_config["render"] = True
    env_config["on_rack"] = False
    env_config["enable_joint_velocity_estimate"] = False
    env_config["isRLGymInterface"] = False
    env_config["robot_model"] = "GO1"
    env_config["motor_control_mode"] = "TORQUE"
    env_config["action_repeat"] = 1
    
    env = QuadrupedGymEnv(**env_config)
    env = JumpingStateMachine(env)

    return env
    
if __name__ == '__main__':
    
    env = build_env()
    sim_steps = env._total_sim_steps
    
    for _ in range(sim_steps):
        action = env._compute_action()
        obs, reward, done, info = env.step(action)
    
    env.close()
    print("end")
    print(env._max_height)
        
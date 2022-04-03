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
        self._step_counter = 0
        self._settling_duration_steps = 1000
        self._couching_duration_steps = 2000
        assert self._couching_duration_steps >= 1000, 'couching duration steps number should be >= 1000'
        self._states = {'settling': 0, 'couching': 1, 'jumping_ground': 2, 'jumping_air': 3, 'landing': 4}
        self._state = self._states['settling']
        self._flying_up_counter = 0
        self._actions = {0: self._settling_action, 1: self._couching_action, 2: self._jumping_explosive_action, 3: self._jumping_flying_action, 4: self._jumping_landing_action}
        self._total_sim_steps = 5000

    def _compute_action(self):
        return self._actions[self._state]()

    def _update_state(self):
        if self._step_counter <= self._settling_duration_steps:
            actual_state = self._states['settling']
        elif self._step_counter <= self._couching_duration_steps:
            actual_state = self._states['couching']
        else:
            if self._all_feet_in_contact():
                actual_state = self._states['jumping_ground']
            else:
                if self._is_landing():
                    actual_state = self._states['landing']
                else:
                    actual_state = self._states['jumping_air']
            
        self._state = actual_state
        
    def _settling_action(self):
        action = np.full(12,0)
        return action
        
    def _couching_action(self):
        action = np.full(12,0)
        return action
    
    def _jumping_explosive_action(self):
        action = np.full(12,0)
        return action
    
    def _jumping_flying_action(self):
        action = np.full(12,0)
        return action
    
    def _jumping_landing_action(self):
        action = np.full(12,0)
        return action
        
    def _all_feet_in_contact(self):
        _, _, _, feetInContactBool = self.env.robot.GetContactInfo()
        return np.all(feetInContactBool)
    
    def _is_landing(self):
        if self._flying_up_counter >= 800:
            return True
        else:
            self._flying_up_counter += 1
            return False
        
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
        
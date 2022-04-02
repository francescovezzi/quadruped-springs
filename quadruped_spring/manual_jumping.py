import time

import numpy as np
from env.quadruped_gym_env import QuadrupedGymEnv


class StateMachine(QuadrupedGymEnv):

    
    def __init__(self, on_rack=False, render=True, enable_springs=True):
        super().__init__(on_rack=on_rack, render=render, enable_springs=enable_springs)

    def temporary_switch_motor_control_mode(mode):
        def _decorator(foo):
            def wrapper(self, *args, **kwargs):
                """Settle robot and add noise to init configuration."""
                # change to 'mode' control mode to set initial position, then set back..
                tmp_save_motor_control_mode_ENV = self._motor_control_mode
                tmp_save_motor_control_mode_ROB = self.robot._motor_control_mode
                self._motor_control_mode = mode
                self.robot._motor_control_mode = mode
                try:
                    tmp_save_motor_control_mode_MOT = self.robot._motor_model._motor_control_mode
                    self.robot._motor_model._motor_control_mode = mode
                except:
                    pass
                foo(self, *args, **kwargs)
                # set control mode back
                self._motor_control_mode = tmp_save_motor_control_mode_ENV
                self.robot._motor_control_mode = tmp_save_motor_control_mode_ROB
                try:
                    self.robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
                except:
                    pass
            return wrapper
        return _decorator
    
    @temporary_switch_motor_control_mode(mode='PD')
    def settle_init_config(self):
        init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
        if self._is_render:
            time.sleep(0.2)
        for _ in range(800):
            self.robot.ApplyAction(init_motor_angles)
            if self._is_render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()

    def _super_decorator(mode):    
        def _decorator(foo):
            def wrapper(self, *args, **kwargs):
                print(f'change temporary to {mode}')
                foo(self, *args, **kwargs)
                print(f'set back to number')
            return wrapper
        return _decorator
    
    # @_super_decorator(mode='pd')
    # def apply_control_dec(self):
    #     print('apply this control')
    


def build_env():
    env_config = {}
    env_config["enable_springs"] = True
    env_config["render"] = True
    env_config["on_rack"] = False

    return StateMachine(**env_config)


if __name__ == '__main__':
    
    env = build_env()
    sim_steps = 1000
    
    env.settle_init_config()

    #obs = env.reset()
    # for i in range(sim_steps):
    #     action = np.random.rand(12) * 2 - 1
    #     # action = np.full(12,0)
    #     obs, reward, done, info = env.step(action)
    env.close()
    print("end")
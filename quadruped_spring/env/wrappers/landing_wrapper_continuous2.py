import gym

from quadruped_spring.utils.timer import Timer


class LandingWrapperContinuous2(gym.Wrapper):
    """
    Wrapper to switch controller when robot starts taking off.
    Dear user please pay attention at the order of the wrapper you are using.
    """

    def __init__(self, env):
        super().__init__(env)
        self._robot_config = self.env.get_robot_config()
        self._landing_action = self.env.get_landing_action()
        self.timer_jumping = Timer(dt=self.env.env_time_step)

    def temporary_switch_motor_control_gain(foo):
        def wrapper(self, *args, **kwargs):
            """Temporary switch motor control gain"""
            if self.env.are_springs_enabled():
                kp = 60.0
                kd = 1.5
            else:
                kp = 60.0
                kd = 1.5
            tmp_save_motor_kp = self.env.robot._motor_model._kp
            tmp_save_motor_kd = self.env.robot._motor_model._kd
            self.env.robot._motor_model._kp = kp
            self.env.robot._motor_model._kd = kd
            ret = foo(self, *args, **kwargs)
            self.env.robot._motor_model._kp = tmp_save_motor_kp
            self.env.robot._motor_model._kd = tmp_save_motor_kd
            return ret

        return wrapper

    # @temporary_switch_motor_control_gain
    def landing_phase(self):
        action = self._landing_action
        done = False
        obs, reward, done, infos = self.env.step(action)
        while not done and self.env.task.get_jumping():
            obs, reward, done, infos = self.env.step(action)

        return obs, reward, done, infos

    def take_off_phase(self, action):
        """Repeat last action until you rech the height peak"""
        done = False
        self.start_jumping_timer()
        while not (self.timer_jumping.time_up() or done):  # episode or timer end
            self.timer_jumping.step_timer()
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def start_jumping_timer(self):
        actual_time = self.env.get_sim_time()
        delta_time = self.env.task.compute_time_for_peak_heihgt()
        self.timer_jumping.reset_timer()
        self.timer_jumping.start_timer(timer_time=actual_time, start_time=actual_time, delta_time=delta_time)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        if self.env.task.get_jumping() and not done:
            obs, reward, done, infos = self.take_off_phase(action)
            # if not done:
            #     _, reward, done, infos = self.landing_phase()

        return obs, reward, done, infos

    def reset(self):
        obs = self.env.reset()

        return obs

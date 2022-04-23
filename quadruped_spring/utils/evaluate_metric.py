import gym
import numpy as np


class EvaluateMetricJumpOnPlace(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.init_metric()
        self._sep = 20
        self.jump_metric_old = 0.0
        self.flag_first = True

    def compute_max_power(self):
        tau = abs(self.env.robot.GetMotorTorques())
        vel = abs(self.env.robot.GetMotorVelocities())

        return max(*(tau * vel))

    def compute_forward_distance(self):
        x, y, _ = self.env.robot.GetBasePosition()
        dx = x - self.x_pos
        dy = y - self.y_pos
        return np.sqrt(dx**2 + dy**2)

    def update_metrics(self):
        self.jump_metric_old = max(self.jump_metric_old, self.jump_metric)
        self.init_metric()

    def init_metric(self):
        self.x_pos, self.y_pos, self.height = self.env.robot.GetBasePosition()
        self.roll, _, self.yaw = abs(self.env.robot.GetBaseOrientationRollPitchYaw())
        self.power = 0.0
        self.penalization_invalid_contact = 0
        self._min_height = self.height
        self._init_height = self.height
        self._landed = False
        self._taking_off = False
        self.jump_metric = 0.0
        self.bounce_counter = 0
        self.all_feet_in_contact = True

    def eval_metric(self):
        _, numInvalidContacts, _, feet_in_contact = self.env.robot.GetContactInfo()
        if numInvalidContacts > 0:
            self.penalization_invalid_contact = -10
        self.power = max(self.power, self.compute_max_power())
        roll, _, yaw = self.env.robot.GetBaseOrientationRollPitchYaw()
        _, _, height = self.env.robot.GetBasePosition()
        self.roll = max(self.roll, abs(roll))
        self.yaw = max(self.yaw, abs(yaw))
        self.height = max(self.height, abs(height))
        self._min_height = min(self._min_height, abs(height))
        if np.all(1 - np.array([feet_in_contact])):
            self._taking_off = True
            self._landed = False
            if self.all_feet_in_contact:
                self.bounce_counter += 1
                self.all_feet_in_contact = False
        if self._taking_off and np.all(np.array([feet_in_contact])):
            self._landed = True
            self._taking_off = False
            self.all_feet_in_contact = True

        self.get_metric()

    def print_first_line_table(self):
        s1 = "ID"
        s2 = "METRIC"
        s3 = "MAX_POWER [w]"
        s4 = "MAX_HEIGHT [m]"
        s5 = "MIN_HEIGHT [m]"
        sep = self._sep
        columns = [s1, s2, s3, s4, s5]
        first_line = ""
        # print('*' * (sep * len(columns) + 4))
        for c in columns:
            first_line += c + " " * (sep - len(c))
        print(first_line)
        return first_line

    def fill_line(self, id):
        columns = [id, f"{self.get_metric():.3f}", f"{self.power:.3f}", f"{self.height:.3f}", f"{self._min_height:.3f}"]
        line = ""
        for c in columns:
            line += str(c) + " " * (self._sep - len(str(c)))
        print(line)
        return line

    def get_metric(self):
        metric = 0
        max_power = self.power
        if self._landed and abs(max_power) >= 0.01:
            max_height_rel = max(self.height - self._init_height, 0)
            rew_dist = 1 / 3 * max_height_rel * np.exp(-self.compute_forward_distance() ** 2 / 0.1)
            rew_roll = 1 / 3 * max_height_rel * np.exp(-self.roll**2 / 0.1)
            rew_yaw = 1 / 3 * max_height_rel * np.exp(-self.yaw**2 / 0.1)

            metric = rew_dist + rew_roll + rew_yaw + max_height_rel * 1000 / (2 * max_power)
            metric += self.penalization_invalid_contact
            metric -= max(self.bounce_counter - 1, 0) * 0.2
        self.jump_metric = max(self.jump_metric, metric)
        return max(-1, self.jump_metric, self.jump_metric_old)

    def print_metric(self):
        print(f"the jump (on place) metric performance amounts to: {self.get_metric():.3f}")
        print(f"the maximum reached height amounts to: {self.height:.3f}")

    def step(self, action):

        if self.flag_first:
            self.flag_first = False
            self.init_metric()

        obs, reward, done, infos = self.env.step(action)
        self.eval_metric()
        if done:
            self.update_metrics()

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()

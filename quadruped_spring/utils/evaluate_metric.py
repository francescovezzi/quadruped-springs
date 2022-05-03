import gym
import numpy as np


class MetricInfo:
    def __init__(self, metric=0.0, power=0.0, height_max=0.0, height_min=0.0, tau_max=0.0):
        self.metric_value = metric
        self.power_max = power
        self.height_max = height_max
        self.height_min = height_min
        self.tau_max = tau_max

    @staticmethod
    def get_metric_values(self):
        return (self.metric_value, self.power_max, self.height_max, self.height_min, self.tau_max)

    def best_metrics(a, b):
        if a.metric_value >= b.metric_value:
            return a
        else:
            return b


class EvaluateMetricJumpOnPlace(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.init_metric()
        self._sep = 20
        self.jump_metric_old = MetricInfo()
        self.flag_first = True

    def compute_max_power(self):
        tau = abs(self.env.robot.GetMotorTorques())
        vel = abs(self.env.robot.GetMotorVelocities())

        return max(*(tau * vel))

    def compute_max_torque(self):
        tau = abs(self.env.robot.GetMotorTorques())
        return max(*tau)

    def compute_forward_distance(self):
        x, y, _ = self.env.robot.GetBasePosition()
        dx = x - self.x_pos
        dy = y - self.y_pos
        return np.sqrt(dx**2 + dy**2)

    def update_metrics(self):
        self.jump_metric_old = MetricInfo.best_metrics(self.jump_metric_old, self.jump_metric)
        self.init_metric()

    def init_metric(self):
        self.x_pos, self.y_pos, self.height = self.env.robot.GetBasePosition()
        self.roll, _, self.yaw = abs(self.env.robot.GetBaseOrientationRollPitchYaw())
        self._landed = False
        self._taking_off = False
        self.jump_metric = MetricInfo(height_max=self.height, height_min=self.height)
        self._init_height = self.height
        self.bounce_counter = 0
        self.all_feet_in_contact = True

    def eval_metric(self):
        _, _, _, feet_in_contact = self.env.robot.GetContactInfo()
        self.jump_metric.power_max = max(self.jump_metric.power_max, self.compute_max_power())
        self.jump_metric.tau_max = max(self.jump_metric.tau_max, self.compute_max_torque())
        roll, _, yaw = self.env.robot.GetBaseOrientationRollPitchYaw()
        _, _, height = self.env.robot.GetBasePosition()
        self.roll = max(self.roll, abs(roll))
        self.yaw = max(self.yaw, abs(yaw))
        self.jump_metric.height_max = max(self.jump_metric.height_max, abs(height))
        self.jump_metric.height_min = min(self.jump_metric.height_min, abs(height))
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
        s6 = "MAX_TORQUE [Nm/rad]"
        s7 = "SPRING"
        sep = self._sep
        columns = [s1, s2, s3, s4, s5, s6, s7]
        first_line = ""
        # print('*' * (sep * len(columns) + 4))
        for c in columns:
            first_line += c + " " * (sep - len(c))
        print(first_line)
        return first_line + "\n"

    def fill_line(self, id):
        metric, power, height_max, height_min, torque_max = self.get_metric().get_metric_values()
        columns = [
            id,
            f"{metric:.3f}",
            f"{power:.3f}",
            f"{height_max:.3f}",
            f"{height_min:.3f}",
            f"{torque_max:.3f}",
            f"{self.env._enable_springs}",
        ]
        line = ""
        for c in columns:
            line += str(c) + " " * (self._sep - len(str(c)))
        print(line)
        return line + "\n"

    def get_metric(self):
        metric = 0
        max_power = self.jump_metric.power_max
        max_torque = self.jump_metric.tau_max
        if self._landed and abs(max_torque) >= 0.01:
            max_height_rel = max(self.jump_metric.height_max - self._init_height, 0)
            rew_dist = 1 / 3 * max_height_rel * np.exp(-self.compute_forward_distance() ** 2 / 0.1)
            rew_roll = 1 / 3 * max_height_rel * np.exp(-self.roll**2 / 0.1)
            rew_yaw = 1 / 3 * max_height_rel * np.exp(-self.yaw**2 / 0.1)

            metric = rew_dist + rew_roll + rew_yaw + max_height_rel * 35 / max_torque + 0.01 * max_power / max_torque
            metric -= max(self.bounce_counter - 1, 0) * 0.2
        self.jump_metric.metric_value = max(self.jump_metric.metric_value, metric)
        if self.env.task_terminated():
            self.jump_metric.metric_value = 0.0
        return MetricInfo.best_metrics(self.jump_metric, self.jump_metric_old)

    def print_metric(self):
        print(f"the jump (on place) metric performance amounts to: {self.get_metric().metric_value:.3f}")
        print(f"the maximum reached height amounts to: {self.get_metric().height_max:.3f}")
        print(f"the minimum reached height amounts to: {self.get_metric().height_min:.3f}")

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

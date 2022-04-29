import numpy as np


class Timer:
    def __init__(self):
        self.timer_started = False

    def start_timer(self, timer_time, start_time, delta_time):
        if not self.timer_started:
            self.timer_started = True
            self.timer_time = timer_time
            self.start_time = start_time
            self.end_time = self.start_time + delta_time
            assert self.start_time <= self.end_time, "timer not feasible"

    def reset_timer(self):
        self.timer_started = False

    def time_up(self):
        return self.timer_time > self.end_time

    def update_time(self, time):
        self.timer_time = time

    def already_started(self):
        return self.timer_started

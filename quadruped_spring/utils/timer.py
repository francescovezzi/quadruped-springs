class Timer:
    def __init__(self, dt):
        """Class to simulate a Timer.

        Args:
            dt (Float): It means how much the timer time should increase by default
                        each time the increase method is called.
        """
        self.timer_started = False
        self.timer_stopped = False
        assert dt >= 0.0, "delta time given should be >= 0"
        self.dt = dt

    def step_timer(self):
        if self.already_started():
            self.timer_time += self.dt
        else:
            raise ValueError("Timer not started yet")

    def start_timer(self, timer_time, start_time, delta_time):
        if not self.timer_started:
            self.timer_started = True
            self.timer_time = timer_time
            self.start_time = start_time
            self.end_time = self.start_time + delta_time
            if not self.timer_stopped:
                self.relative_timer_time = self.timer_time - self.start_time
            assert self.start_time <= self.end_time, "timer not feasible"

    def stop_timer(self):
        self.timer_stopped = True
        return self.relative_timer_time

    def reset_timer(self):
        self.timer_started = False
        self.timer_stopped = False

    def time_up(self):
        if self.timer_started:
            return self.timer_time > self.end_time
        else:
            return self.timer_started

    def update_time(self, time):
        self.timer_time = time

    def already_started(self):
        return self.timer_started

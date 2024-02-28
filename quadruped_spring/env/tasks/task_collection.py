from quadruped_spring.env.tasks import robot_tasks as rt
from quadruped_spring.utils.base_collection import CollectionBase

class TaskCollection(CollectionBase):
    """Utility to collect all the implemented robot tasks."""

    def __init__(self):
        super().__init__()
        self.JUMPING_IN_PLACE = rt.JumpingInPlace
        self.JUMPING_FORWARD = rt.JumpingForward
        self.NO_TASK = rt.NoTask
        self.JIP_DEMO = rt.JumpingDemoInPlace
        self.JIP_PPO = rt.JumpingInPlacePPO
        self.JF_DEMO = rt.JumpingDemoForward
        self.JF_PPO = rt.JumpingForwardPPO
        self.JIP_PPO_HP = rt.JumpingInPlacePPOHP
        self.JF_PPO_HP = rt.JumpingForwardPPOHP
        self._element_type = "task"
        self._dict = {
            "JUMPING_FORWARD": self.JUMPING_FORWARD,
            "JUMPING_IN_PLACE": self.JUMPING_IN_PLACE,
            "NO_TASK": self.NO_TASK,
            "JUMPING_IN_PLACE_DEMO": self.JIP_DEMO,
            "JUMPING_IN_PLACE_PPO": self.JIP_PPO,
            "JUMPING_FORWARD_DEMO": self.JF_DEMO,
            "JUMPING_FORWARD_PPO": self.JF_PPO,
            "JUMPING_IN_PLACE_PPO_HP": self.JIP_PPO_HP,
            "JUMPING_FORWARD_PPO_HP": self.JF_PPO_HP,
            "BACKFLIP": rt.BackFlip,
            "BACKFLIP_DEMO": rt.BackflipDemo,
            "CONTINUOUS_JUMPING_FORWARD_PPO": rt.ContinuousJumpingForwardPPO,
            "BACKFLIP_PPO": rt.BackflipPPO,
            "CONTINUOUS_JUMPING_FORWARD": rt.JumpingForwardContinuous,
            "CONTINUOUS_JUMPING_FORWARD2": rt.JumpingForwardContinuous2,
            "CONTINUOUS_JUMPING_FORWARD3": rt.JumpingForwardContinuous3,
            "CONTINUOUS_JUMPING_FORWARD_DEMO": rt.JumpingContinuousDemoForward,
        }

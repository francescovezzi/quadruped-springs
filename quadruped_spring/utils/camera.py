import time
from enum import Enum

import numpy as np


class Camera:
    def __init__(self, env, pybullet_client):
        self._env = env
        self._pybullet_client = pybullet_client
        self.set_camera_variables()
        self._init_rendering_variables()

    def set_camera_variables(self):
        self._cam_dist = 1.3
        self._cam_yaw = 20
        self._cam_pitch = -20
        self._fov = 60

    def reset(self):
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])

    def _init_rendering_variables(self):
        self._last_frame_time = 0.0

        self._render_width = 1440
        self._render_height = 1080

        # get rid of visualizer things
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.compute_camera_pos()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=self._fov, aspect=float(self._render_width) / self._render_height, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def compute_camera_pos(self):
        return self._env.robot.GetBasePosition()

    def _render_step_helper(self):
        """Helper to configure the visualizer camera during step()."""
        # Sleep, otherwise the computation takes less time than real time,
        # which will make the visualization like a fast-forward video.
        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        # time_to_sleep = self._action_repeat * self._time_step - time_spent
        time_to_sleep = self._env.sim_time_step - time_spent
        if time_to_sleep > 0 and (time_to_sleep < self._env.sim_time_step):
            time.sleep(time_to_sleep)

        base_pos = self.compute_camera_pos()
        camInfo = self._pybullet_client.getDebugVisualizerCamera()
        # curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]
        # targetPos = [0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1], curTargetPos[2]]
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)


class FixedCamera(Camera):
    def __init__(self, env, pybullet_client, camera_pos=None):
        super().__init__(env, pybullet_client)
        self.camera_pos = camera_pos

    def compute_camera_pos(self):
        return self.camera_pos


class BackFlipCamera(FixedCamera):
    def __init__(self, env, pybullet_client):
        super().__init__(env, pybullet_client, camera_pos=[-0.55, 0.0, 0.6])

    def set_camera_variables(self):
        self._cam_dist = 1.3
        self._cam_yaw = 0
        self._cam_pitch = -6
        self._fov = 80
        
class ContinuousJumpingForwardCamera(Camera):
    def __init__(self, env, pybullet_client):
        super().__init__(env, pybullet_client)
        
    def set_camera_variables(self):
        self._cam_dist = 1.3
        self._cam_yaw = 10
        self._cam_pitch = -8
        self._fov = 80

class CameraModes(Enum):
    CLASSIC = Camera
    BACKFLIP = BackFlipCamera
    CONTINUOUS_JUMPING_FORWARD = ContinuousJumpingForwardCamera

def make_camera(env, mode="Classic"):
    camera_obj = CameraModes[mode.upper()].value
    pybullet_client = env._pybullet_client
    return camera_obj(env, pybullet_client)

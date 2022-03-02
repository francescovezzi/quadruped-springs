from gym.envs.registration import register

register(
    id="QuadrupedSpring-v0",
    entry_point="quadruped_spring.env.quadruped_gym_env:QuadrupedGymEnv",
    kwargs={
        "motor_control_mode": "CARTESIAN_PD",
        "task_env": "LR_COURSE_TASK",
        "observation_space_mode": "LR_COURSE_OBS",
    },
)


# register(
#     id="QuadrupedSpringTorques-v0",
#     entry_point="quadruped_spring.env.quadruped_gym_env:QuadrupedGymEnv",
#     kwargs={
#         "motor_control_mode": "TORQUE",
#         "task_env": "LR_COURSE_TASK",
#         "observation_space_mode": "LR_COURSE_OBS",
#     },
# )

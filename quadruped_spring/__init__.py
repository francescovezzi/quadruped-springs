from gym.envs.registration import register

register(
    id="QuadrupedSpring-v0",
    entry_point="quadruped_spring.env.quadruped_gym_env:QuadrupedGymEnv",
    kwargs={
        "motor_control_mode": "PD",
        "task_env": "JUMPING_IN_PLACE",
        "observation_space_mode": "ARS_HEIGHT",
        "action_space_mode": "SYMMETRIC",
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

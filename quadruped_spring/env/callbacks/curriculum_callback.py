from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from quadruped_spring.env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.env.wrappers.curriculum_wrapper import CurriculumWrapper

import warnings


def callable_env(kwargs):
    def aux():
        env = QuadrupedGymEnv(**kwargs)
        env = CurriculumWrapper(env)
        env = LandingWrapper(env)
        env = ObsFlatteningWrapper(env)
        return env
    return aux

class CurriculumCallback(EventCallback):
    """
    Increase the curriculum level during training.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=1):
        super().__init__(None, verbose)
        
        self.n_eval_env = 4
        self.eval_freq = 4_000_000
        self.n_eval_episodes = 10
        self.reward_threshold = 0.75
        self.level_step = 0.05
        self.deterministic = True
        
        self.eval_freq = 100
        self.n_eval_episodes = 2
        self.reward_threshold = -20
        
    def _init_callback(self) -> None:
        env_config = self.training_env.env_method("get_env_kwargs", indices=0)[0]
        eval_env = callable_env(env_config)
        eval_env = make_vec_env(eval_env, n_envs=self.n_eval_env)
    
        normalize_kwargs = dict(training = False, norm_reward = True)
        eval_env = VecNormalize(eval_env, **normalize_kwargs)
        
        self.eval_env = eval_env
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
            
        self.training_env.env_method("print_curriculum_info", indices=0)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )
            mean_reward, std_reward = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=False,
                    deterministic=True,
                    return_episode_rewards=False,
                    warn=True,
                )
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")

            if mean_reward > self.reward_threshold:
                self.training_env.env_method("increase_curriculum_level", self.level_step, indices=range(self.training_env.num_envs))
                self.eval_env.env_method("increase_curriculum_level", self.level_step, indices=range(self.eval_env.num_envs))
                self.training_env.env_method("print_curriculum_info", indices=0)

        return True
        
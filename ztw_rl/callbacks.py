from pathlib import Path
import math as m
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import neptune.new as neptune
import numpy as np
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common import logger


class NeptuneTrainCallback(BaseCallback):
    # TODO after https://github.com/DLR-RM/stable-baselines3/issues/109 is resolved
    # implement this as stable baselines3 logger and switch the logger when using neptune
    # (logger.dump() in on_policy_algorithm.py causes this callback to miss some statistics)
    # then both these callbacks would be redundant (model classes and EvalCallback both write into logger)
    def __init__(
        self,
        neptune_run: neptune.run.Run,
        total_timesteps: int = -1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.run = neptune_run
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        # log progress
        self.run['progress'] = self.num_timesteps / self.total_timesteps
        return True

    def _on_rollout_end(self) -> None:
        # log logger contents
        log_dict = logger.get_log_dict()
        for k, v in log_dict.items():
            self.run[k].log(v, self.num_timesteps)
        pass

    def _on_training_end(self) -> None:
        self.run['progress'] = 1.0


class NeptuneEvalCallback(EventCallback):
    """
    Callback for evaluating an agent and saving the results into neptune.
    """
    def __init__(
        self,
        neptune_run: neptune.run.Run,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        best_model_save_path: Path = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        self.run = neptune_run
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -m.inf
        self.last_mean_reward = -m.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            self.best_model_save_path.mkdir(exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        # VecEnv: unpack
        if not isinstance(info, dict):
            info = info[0]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            # Add to neptune
            self.run['eval/mean_reward'].log(mean_reward, self.num_timesteps)
            self.run['eval/std_reward'].log(std_reward, self.num_timesteps)
            self.run['eval/mean_ep_length'].log(mean_ep_length, self.num_timesteps)
            self.run['eval/std_ep_length'].log(std_ep_length, self.num_timesteps)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                self.run['eval/success_rate'].log(success_rate, self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f'New best mean reward at timestep {self.num_timesteps}')
                if self.best_model_save_path is not None:
                    save_path = self.best_model_save_path / 'best_model.zip'
                    self.model.save(save_path)
                    self.run['best_model'].upload(str(save_path))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import gym
import neptune.new as neptune
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.preprocessing import (
    get_obs_shape, is_image_space, is_image_space_channels_first)
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from callbacks import NeptuneEvalCallback, NeptuneTrainCallback
from profiler import profile_ee
from wrappers import make_sticky_atari_env, make_sticky_vec_env
from ztw import EEModel, InternalClassifier


def is_atari(env_id: str) -> bool:
    return "AtariEnv" in gym.envs.registry.env_specs[env_id].entry_point


def make_envs(env_id: str,
              n_envs: int,
              frame_stack: Optional[int],
              monitor_dir: Optional[Path] = None,
              eval_monitor_dir: Optional[Path] = None) -> Tuple[VecEnv, VecEnv]:
    # make_atari/vec_env will create a DummyVecEnv
    if is_atari(env_id):
        env = make_sticky_atari_env(env_id, n_envs=n_envs, monitor_dir=monitor_dir)
        eval_env = make_sticky_atari_env(env_id, n_envs=1, monitor_dir=eval_monitor_dir)
    else:
        env = make_sticky_vec_env(env_id, n_envs=n_envs, monitor_dir=monitor_dir)
        eval_env = make_sticky_vec_env(env_id, n_envs=1, monitor_dir=eval_monitor_dir)
    if frame_stack is not None:
        env = VecFrameStack(env, n_stack=frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
        eval_env = VecTransposeImage(eval_env)
    # TODO optionally add env seeds
    # if seed is not None:
    #     env.seed(seed + rank)
    #     env.action_space.seed(seed + rank)
    # TODO optionally add VecNormalize with a flag?
    # ...
    return env, eval_env


def train(s: SimpleNamespace, results_dir: Path, neptune_run: neptune.run.Run = None) -> BaseAlgorithm:
    monitor_dir = results_dir / 'monitor'
    monitor_dir.mkdir()
    eval_monitor_dir = results_dir / 'eval_monitor'
    eval_monitor_dir.mkdir()
    if neptune_run is not None:
        tensorboard_dir = None
    else:
        tensorboard_dir = results_dir / 'tb'
        tensorboard_dir.mkdir(exist_ok=True)
        tensorboard_dir = str(tensorboard_dir)
    frame_stack = None if not hasattr(s, 'frame_stack') else s.frame_stack
    env, eval_env = make_envs(s.env_id, s.n_envs, frame_stack, monitor_dir, eval_monitor_dir)
    eval_freq = max(s.eval_freq // s.n_envs, 1)
    callbacks = []
    if neptune_run is not None:
        callbacks.append(
            NeptuneEvalCallback(neptune_run,
                                eval_env,
                                best_model_save_path=results_dir,
                                eval_freq=eval_freq,
                                deterministic=s.deterministic,
                                render=False))
        callbacks.append(NeptuneTrainCallback(neptune_run, s.timesteps))
    else:
        callbacks.append(
            EvalCallback(eval_env,
                         best_model_save_path=results_dir,
                         log_path=results_dir,
                         eval_freq=eval_freq,
                         deterministic=s.deterministic,
                         render=False))
    model_class = s.algo
    model = model_class(env=env, tensorboard_log=tensorboard_dir, **s.algo_args)
    model.learn(total_timesteps=s.timesteps, callback=callbacks)
    # save files
    policy_path = results_dir / 'policy'
    save_path = results_dir / 'model'
    settings_path = results_dir / 'settings'
    model.save(policy_path)
    ztw_model = EEModel(model, [], InternalClassifier, 'sdn')
    torch.save(ztw_model, save_path)
    torch.save(s, settings_path)
    # profile basic model/policy (total ops etc.) and save results
    total_ops, total_params = profile_ee(ztw_model, get_obs_shape(eval_env.observation_space))
    total_ops, total_params = total_ops[0], total_params[0]
    ops_path = results_dir / 'total_ops'
    params_path = results_dir / 'total_params'
    torch.save(total_ops, ops_path)
    torch.save(total_params, params_path)

    if neptune_run is not None:
        neptune_run['policy'].upload(str(policy_path))
        neptune_run['model'].upload(str(save_path))
        neptune_run['settings_file'].upload(str(settings_path))
        neptune_run['total_ops'].upload(str(ops_path))
        neptune_run['total_params'].upload(str(params_path))

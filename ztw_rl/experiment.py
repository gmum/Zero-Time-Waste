from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.ppo import PPO

from profiler import profile_ee
from train import make_envs, train
from utils import get_run_by_name, init_neptune
from ztw import EEModel, REModel, eval_ztw, train_on_policy

def next_run_number(root_dir: Path, run_name: str) -> int:
    try_number = 0
    while (root_dir / f'{run_name}#{try_number}').exists():
        try_number += 1
    return try_number


def policy_train(settings: SimpleNamespace, run_number: Optional[int] = None, enable_neptune: bool = False) -> Path:
    root_dir = Path('runs')
    run_number = next_run_number(root_dir, settings.run_name) if run_number is None else run_number
    settings.run_name = f'{settings.run_name}#{run_number}'
    results_dir = root_dir / settings.run_name
    results_dir.mkdir(parents=True, exist_ok=False)
    neptune_run = init_neptune(settings) if enable_neptune else None
    train(settings, results_dir, neptune_run)
    return results_dir


def ee_train(parent_results_dir: Path,
             overwrite_settings: SimpleNamespace,
             model_type: str = 'sdn',
             run_number: Optional[int] = None,
             enable_neptune: bool = False) -> Path:
    # TODO handle loading from neptune?
    root_dir = parent_results_dir.parent
    parent_settings_path = parent_results_dir / 'settings'
    settings = torch.load(parent_settings_path)
    # add or overwrite some settings
    settings.__dict__.update(overwrite_settings.__dict__)
    settings.type = model_type
    settings.parent_run_name = settings.run_name
    run_name = f'{settings.parent_run_name}_{settings.type}'
    run_number = next_run_number(root_dir, run_name) if run_number is None else run_number
    settings.run_name = f'{run_name}#{run_number}'

    neptune_run = init_neptune(settings) if enable_neptune else None
    results_dir = parent_results_dir.parent / settings.run_name
    results_dir.mkdir()

    model = settings.algo.load(parent_results_dir / 'policy')
    env, eval_env = make_envs(settings.env_id, settings.n_envs, settings.frame_stack)

    ee_policy = EEModel(model, settings.heads_at, settings.ic_class, model_type, ic_width_multiplier=settings.ic_width_multiplier)
    train_on_policy(settings, ee_policy, env, neptune_run)

    save_path = results_dir / 'model'
    settings_path = results_dir / 'settings'
    torch.save(ee_policy, save_path)
    torch.save(settings, settings_path)

    total_ops, total_params = profile_ee(ee_policy, get_obs_shape(eval_env.observation_space))
    ops_path = results_dir / 'total_ops'
    params_path = results_dir / 'total_params'
    torch.save(total_ops, ops_path)
    torch.save(total_params, params_path)

    if neptune_run is not None:
        neptune_run['model'].upload(str(save_path))
        neptune_run['settings_file'].upload(str(settings_path))
        neptune_run['total_ops'].upload(str(ops_path))
        neptune_run['total_params'].upload(str(params_path))
    return results_dir


def re_train(parent_results_dir: Path,
             overwrite_settings: SimpleNamespace,
             run_number: Optional[int] = None,
             enable_neptune: bool = False) -> Path:
    # TODO handle loading from neptune?
    root_dir = parent_results_dir.parent
    parent_settings_path = parent_results_dir / 'settings'
    settings = torch.load(parent_settings_path)
    # add or overwrite some settings
    settings.__dict__.update(overwrite_settings.__dict__)
    settings.parent_run_name = settings.run_name
    run_name = f'{settings.parent_run_name}_rensb'
    run_number = next_run_number(root_dir, run_name) if run_number is None else run_number
    settings.run_name = f'{run_name}#{run_number}'

    neptune_run = init_neptune(settings) if enable_neptune else None
    results_dir = parent_results_dir.parent / settings.run_name
    results_dir.mkdir()

    model = torch.load(parent_results_dir / 'model', map_location=get_device())
    env, eval_env = make_envs(settings.env_id, settings.n_envs, settings.frame_stack)

    re_policy = REModel(model)
    train_on_policy(settings, re_policy, env, neptune_run)

    save_path = results_dir / 'model'
    settings_path = results_dir / 'settings'
    torch.save(re_policy, save_path)
    torch.save(settings, settings_path)

    total_ops, total_params = profile_ee(re_policy, get_obs_shape(eval_env.observation_space))
    ops_path = results_dir / 'total_ops'
    params_path = results_dir / 'total_params'
    torch.save(total_ops, ops_path)
    torch.save(total_params, params_path)

    if neptune_run is not None:
        neptune_run['model'].upload(str(save_path))
        neptune_run['settings_file'].upload(str(settings_path))
        neptune_run['total_ops'].upload(str(ops_path))
        neptune_run['total_params'].upload(str(params_path))
    return results_dir


def eval_and_save(results_dir: Path, thresholds: List[float], enable_neptune: bool = False):
    model_path = results_dir / 'model'
    settings_path = results_dir / 'settings'
    model = torch.load(model_path, map_location=get_device())
    settings = torch.load(settings_path)
    env, eval_env = make_envs(settings.env_id, 8, settings.frame_stack)
    threshold_results = {}
    for t in thresholds:
        mean_reward, std_reward, chosen_ics = eval_ztw(model,
                                                       eval_env,
                                                       deterministic=settings.deterministic,
                                                       conf_threshold=t)
        threshold_results[t] = (mean_reward, std_reward, chosen_ics)
    save_path = results_dir / 'eval_results'
    torch.save(threshold_results, save_path)
    if enable_neptune:
        neptune_run = get_run_by_name(settings.run_name)
        if neptune_run is not None:
            try:
                neptune_run['eval_results'].upload(str(save_path))
            except:
                pass


def eval_and_save_ics(results_dir: Path, enable_neptune: bool = False):
    model_path = results_dir / 'model'
    settings_path = results_dir / 'settings'
    model = torch.load(model_path, map_location=get_device())
    settings = torch.load(settings_path)
    env, eval_env = make_envs(settings.env_id, 8, settings.frame_stack)
    ic_results = []
    for i in range(len(model.ics) + 1):
        mean_reward, std_reward, chosen_ics = eval_ztw(model,
                                                       eval_env,
                                                       deterministic=settings.deterministic,
                                                       conf_threshold=0.0,
                                                       stop_on_ic=i)
        ic_results.append((mean_reward, std_reward, chosen_ics))
    save_path = results_dir / 'eval_ics'
    torch.save(ic_results, save_path)
    if enable_neptune:
        neptune_run = get_run_by_name(settings.run_name)
        if neptune_run is not None:
            try:
                neptune_run['eval_ics'].upload(str(save_path))
            except:
                pass

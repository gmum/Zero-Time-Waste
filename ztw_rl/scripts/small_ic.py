import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))

import torch
import torch.nn as nn
from experiment import (ee_train, eval_and_save, eval_and_save_ics,
                        policy_train, re_train)
from stable_baselines3.ppo.ppo import PPO
from ztw import SDNPool

from scripts.common import BigNatureCNN, NUM_THRESHOLDS, START_LINSPACE, SmallerInternalClassifier, StandardInternalClassifier


def lr_schedule(current: int, last_batch: int):
    if current < 0.1 * last_batch:
        return 1e-2
    elif current < 0.75 * last_batch:
        return 1e-3
    else:
        return 1e-4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', type=str)
    args = parser.parse_args()
    enable_neptune = True
    # TRAIN POLICY
    settings = SimpleNamespace()
    settings.algo = PPO
    settings.env_id = args.env_id
    settings.algo_args = {}
    settings.algo_args['policy'] = 'CnnPolicy'
    settings.algo_args['learning_rate'] = 2.5e-4
    settings.algo_args['n_steps'] = 128
    settings.algo_args['batch_size'] = 256
    settings.algo_args['n_epochs'] = 4
    settings.algo_args['clip_range'] = 0.1
    settings.algo_args['ent_coef'] = 0.01
    settings.algo_args['vf_coef'] = 0.5
    settings.frame_stack = 4
    settings.n_envs = 8
    settings.timesteps = 10**7
    settings.run_name = f'kld_small_IC_ppo_{settings.env_id}'
    settings.n_evals_episodes = 5
    settings.eval_freq = 10000
    settings.deterministic = True
    results_dir = policy_train(settings, enable_neptune=enable_neptune)
    # TRAIN SDN
    update_settings = SimpleNamespace()
    update_settings.batch_size = 64
    update_settings.epochs = 5
    update_settings.timesteps = 10**6
    update_settings.optimizer_class = torch.optim.Adam
    update_settings.lamb = 0.0  # 1 is CE, 0 is KLD
    update_settings.ic_lr = lr_schedule
    update_settings.heads_at = [0, 1]
    update_settings.ic_width_multiplier = 1.0
    update_settings.ic_class = SmallerInternalClassifier
    sdn_results_dir = ee_train(results_dir, update_settings, 'sdn', enable_neptune=enable_neptune)
    # TRAIN WITH STACKING
    stacking_results_dir = ee_train(results_dir, update_settings, 'stacking', enable_neptune=enable_neptune)
    # TRAIN RUNNING ENSEMBLES
    update_settings = SimpleNamespace()
    update_settings.batch_size = 128
    update_settings.epochs = 5
    update_settings.timesteps = 10**6
    update_settings.optimizer_class = torch.optim.Adam
    update_settings.lamb = 0.0  # 1 is CE, 0 is KLD
    update_settings.ic_lr = lr_schedule
    rensb_results_dir = re_train(stacking_results_dir, update_settings, enable_neptune=enable_neptune)
    # EVALUATE
    eval_and_save(results_dir, [0.0], enable_neptune=enable_neptune)
    thresholds = torch.linspace(START_LINSPACE, 1.0, steps=NUM_THRESHOLDS).tolist()
    eval_and_save(sdn_results_dir, thresholds, enable_neptune=enable_neptune)
    eval_and_save_ics(sdn_results_dir, enable_neptune=enable_neptune)
    eval_and_save(stacking_results_dir, thresholds, enable_neptune=enable_neptune)
    eval_and_save_ics(stacking_results_dir, enable_neptune=enable_neptune)
    eval_and_save(rensb_results_dir, thresholds, enable_neptune=enable_neptune)
    eval_and_save_ics(rensb_results_dir, enable_neptune=enable_neptune)

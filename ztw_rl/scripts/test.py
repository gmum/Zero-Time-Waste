import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))

import torch
import torch.nn as nn
from experiment import ee_train, eval_and_save, policy_train, re_train
from stable_baselines3.ppo.ppo import PPO
from ztw import SDNPool

from scripts.common import BigNatureCNN, StandardInternalClassifier



def lr_schedule(current: int, last_batch: int):
    if current < 0.1 * last_batch:
        return 1e-2
    elif current < 0.75 * last_batch:
        return 1e-3
    else:
        return 1e-4


if __name__ == '__main__':
    enable_neptune = False
    print(f'TRAIN POLICY')
    settings = SimpleNamespace()
    settings.algo = PPO
    settings.env_id = 'MsPacman-v0'
    settings.algo_args = {}
    settings.algo_args['policy'] = 'CnnPolicy'
    settings.algo_args['policy_kwargs'] = {'features_extractor_class': BigNatureCNN}
    settings.algo_args['learning_rate'] = 2.5e-4
    settings.algo_args['n_steps'] = 128
    settings.algo_args['batch_size'] = 256
    settings.algo_args['n_epochs'] = 4
    settings.algo_args['clip_range'] = 0.1
    settings.algo_args['ent_coef'] = 0.01
    settings.algo_args['vf_coef'] = 0.5
    settings.frame_stack = 4
    settings.n_envs = 8
    settings.timesteps = 10**2
    settings.run_name = f'test_ppo_{settings.env_id}'
    
    settings.n_evals_episodes = 5
    settings.eval_freq = 10000
    settings.deterministic = True
    results_dir = policy_train(settings, enable_neptune=enable_neptune)
    print(f'TRAIN SDN')
    update_settings = SimpleNamespace()
    update_settings.batch_size = 64
    update_settings.epochs = 5
    update_settings.timesteps = 10**2
    update_settings.optimizer_class = torch.optim.Adam
    update_settings.lamb = 1.0  # 1 is CE, 0 is KLD
    # update_settings.lamb = 0.0
    update_settings.ic_lr = lr_schedule
    update_settings.heads_at = [0, 1]
    update_settings.ic_width_multiplier = 0.5
    update_settings.ic_class = StandardInternalClassifier
    sdn_results_dir = ee_train(results_dir, update_settings, 'sdn', enable_neptune=enable_neptune)
    print(f'TRAIN WITH STACKING')
    stacking_results_dir = ee_train(results_dir, update_settings, 'stacking', enable_neptune=enable_neptune)
    print(f'TRAIN RUNNING ENSEMBLES')
    update_settings = SimpleNamespace()
    update_settings.batch_size = 128
    update_settings.epochs = 5
    update_settings.timesteps = 10**2
    update_settings.optimizer_class = torch.optim.Adam
    update_settings.lamb = 1.0  # 1 is CE, 0 is KLD
    # update_settings.lamb = 0.0
    update_settings.ic_lr = lr_schedule
    rensb_results_dir = re_train(stacking_results_dir, update_settings, enable_neptune=enable_neptune)
    print(f'EVALUATE')
    eval_and_save(results_dir, [0.0], enable_neptune=enable_neptune)
    thresholds = torch.linspace(0.0, 1.0, steps=2).tolist()
    eval_and_save(results_dir, [0.0], enable_neptune=enable_neptune)
    eval_and_save(sdn_results_dir, thresholds, enable_neptune=enable_neptune)
    eval_and_save(stacking_results_dir, thresholds, enable_neptune=enable_neptune)
    eval_and_save(rensb_results_dir, thresholds, enable_neptune=enable_neptune)

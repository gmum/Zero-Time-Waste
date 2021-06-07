from types import SimpleNamespace
from typing import Optional

import neptune.new as neptune
import torch
from sb3_contrib import QRDQN, TQC
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "qrdqn": QRDQN,
    "tqc": TQC,
}


def init_neptune(s: SimpleNamespace) -> neptune.run.Run:
    neptune_run = neptune.init(source_files=['*.py', '**/*.py'])
    neptune_run['name'] = s.run_name
    neptune_run['settings'] = s
    if hasattr(s, 'parent'):
        neptune_run['parent'] = s.parent
    return neptune_run


def get_run_by_id(run_id: str) -> Optional[neptune.run.Run]:
    project = neptune.get_project()
    runs = project.fetch_runs_table().to_runs()
    for run in runs:
        run_id = run['sys/id'].get()
        if run_id == run_id:
            return run


def get_run_by_name(name: str) -> Optional[neptune.run.Run]:
    project = neptune.get_project()
    runs = project.fetch_runs_table().to_runs()
    for run in runs:
        run_name = run['name'].get()
        if run_name == name:
            return run


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g['lr'] = lr

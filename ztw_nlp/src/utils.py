import os
import random

import numpy as np
import torch.utils.data

from src.args import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_run_name(config: Config) -> str:
    base = (
        f"{config.base_model}"
        f"-{config.ee_model}"
        f"-{config.task}"
        f"-{config.seed}"
        f"-epochs_{config.max_epochs}"
        f"-lr_{config.lr}"
    )
    if config.patience is not None:
        base += f"-patience_{config.patience}"
    if config.ensembling:
        base += f"-e_lr_{config.ensembling_lr}" f"-e_epochs_{config.ensembling_epochs}"
    return base

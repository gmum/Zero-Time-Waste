import argparse
import base64
import hashlib
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Union, Tuple, List, Dict, Set, FrozenSet

import torch
import wandb

from common import MODEL_NAME_MAP

device = None


def get_device() -> torch.device:
    global device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
        logging.info(f'{torch.cuda.device_count()} GPUs. Using: {device}')
    return device


def get_loader(data: torch.utils.data.Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 8,
               pin: bool = True) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset=data,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       pin_memory=pin,
                                       num_workers=num_workers)


def make_hashable(o: Union[Tuple, List, Dict, Set, FrozenSet]) -> Union[Tuple, List, Dict, Set, FrozenSet]:
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))
    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))
    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))
    return o


def make_hash_sha256(o: Union[Tuple, List, Dict, Set, FrozenSet]) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b32encode(hasher.digest()).decode()


def generate_run_name(args: Namespace) -> Tuple[str, str]:
    # Properties from the config that are to be included into hashing.
    hashed_keys = ['model_class', 'model_args', 'dataset', 'init_fun', 'batch_size',
                   'loss_type', 'loss_args', 'optimizer_class', 'optimizer_args',
                   'scheduler_class', 'scheduler_args', 'epochs', 'mixup_alpha']
    if args.base_on is not None:
        hashed_keys.extend(['base_on', 'sequentially', 'with_backbone', 'auxiliary_loss_type',
                            'auxiliary_loss_weight'])
    if args.k_fractions is not None:
        hashed_keys.extend(['k_fractions'])
    hashed_flags = {k: v for k, v in vars(args).items() if k in hashed_keys and v is not None}
    short_hash = make_hash_sha256(hashed_flags)[:8]
    exp_name = f'{args.dataset}_{args.model_class}_{short_hash}'
    run_name = f'{exp_name}_{args.exp_id}'
    logging.info(f'Generated run name: {run_name}')
    return exp_name, run_name


def get_run_id(run_name: str):
    api = wandb.Api()
    entity = os.environ['WANDB_ENTITY']
    project = os.environ['WANDB_PROJECT']
    retrieved_runs = api.runs(f'{entity}/{project}', filters={'display_name': run_name})
    logging.info(f'Retrieved {len(retrieved_runs)} for run_name: {run_name}')
    assert len(retrieved_runs) <= 1, f'retrieved_runs: {retrieved_runs}'
    if len(retrieved_runs) == 1:
        return retrieved_runs[0].id


def load_state(args: Namespace, state_path: Path, device: Union[torch.device, str] = get_device()):
    if state_path.exists() and state_path.is_file():
        state = torch.load(state_path, map_location=device)
        logging.info(f'Loaded state from local path {str(state_path)}')
        return state
    elif args.use_wandb:
        try:
            wandb_checkpoint = wandb.restore(str(state_path))
            if wandb_checkpoint is not None:
                state = torch.load(wandb_checkpoint.name, map_location=device)
                logging.info(f'Loaded state from W&B')
                return state
        except ValueError:
            pass
    else:
        raise FileNotFoundError('Cannot find the state file')


def save_state(args: Namespace, state: Dict, state_path: Path):
    tmp_state_path = state_path.parent / f'{state_path.name}.tmp'
    torch.save(state, tmp_state_path)
    logging.info(f'Saved state to local path {str(state_path)}')
    tmp_state_path.replace(state_path)
    tmp_state_path.unlink(missing_ok=True)
    if args.use_wandb:
        wandb.save(str(state_path))
        logging.info(f'Saved state to W&B')


def retrieve_state(args: Namespace, run_name: str, device: Union[torch.device, str] = get_device()):
    base_state_path = args.runs_dir / run_name / f'state.pth'
    if base_state_path.exists() and base_state_path.is_file():
        state = torch.load(base_state_path, map_location=device)
    elif args.use_wandb:
        api = wandb.Api()
        entity = os.environ['WANDB_ENTITY']
        project = os.environ['WANDB_PROJECT']
        retrieved_runs = api.runs(f'{entity}/{project}', filters={'display_name': run_name})
        assert len(retrieved_runs) <= 1, f'retrieved_runs: {retrieved_runs}'
        assert len(retrieved_runs) > 0, f'Run {run_name} not found'
        run = retrieved_runs[0]
        run.file('state.pth').download(root=str(base_state_path.parent), replace=True)
        state = torch.load(base_state_path, map_location=device)
    else:
        raise FileNotFoundError('Cannot find the state file')
    return state


def recreate_base_model(args: argparse.Namespace, base_on: str, base_exp_id: str):
    base_states = []
    base_args = []
    base_model_args = []
    while base_on is not None:
        base_state = retrieve_state(args, f'{base_on}_{base_exp_id}', 'cpu')
        base_arg = base_state['args']
        base_model_arg = json.loads(base_arg.model_args)
        base_states.append(base_state)
        base_args.append(base_arg)
        base_model_args.append(base_model_arg)
        base_on = base_arg.base_on if hasattr(base_arg, 'base_on') else None
        base_exp_id = base_arg.exp_id
    for base_arg, model_arg in zip(reversed(base_args), reversed(base_model_args)):
        if base_arg.base_on is None:
            base_model = MODEL_NAME_MAP[base_arg.model_class](**model_arg)
        else:
            base_model = MODEL_NAME_MAP[base_arg.model_class](base_model, **model_arg)
    base_model.load_state_dict(base_states[0]['model_state'])
    return base_model, base_args[-1]


def load_model(args: argparse.Namespace, run_name: str):
    state = retrieve_state(args, run_name, get_device())
    model_class = state['args'].model_class
    model_args = json.loads(state['args'].model_args)
    if state['args'].base_on is not None:
        base_model, _ = recreate_base_model(args, state['args'].base_on, state['args'].exp_id)
        model = MODEL_NAME_MAP[model_class](base_model, **model_args).to(get_device())
    else:
        model = MODEL_NAME_MAP[model_class](**model_args).to(get_device())
    model.load_state_dict(state['model_state'])
    return model, state

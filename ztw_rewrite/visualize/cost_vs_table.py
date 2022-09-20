import argparse
import logging
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import seaborn as seaborn
import torch
from matplotlib import pyplot as plt

from datasets import DATASETS_NAME_MAP
from eval import get_preds_earlyexiting, evaluate_earlyexiting_calibration, evaluate_earlyexiting_ood_detection, \
    get_preds, evaluate_calibration, evaluate_ood_detection
from utils import load_model, retrieve_state, get_loader
from visualize.cost_vs_plot import parser, PRETTY_NAME_DICT, compute_means_and_stds


def generate_score_eff_table(f, core_stats, point_stats, ee_stats, name_dict, param):
    cost_fractions = [0.25, 0.5, 0.75, 1.0, float('inf')]
    # base_model_run_name = next(iter(core_stats.keys())) # TODO
    assert len(core_stats) > 0
    base_model_run_name = next(iter(core_stats.keys()))
    cost_100 = core_stats[base_model_run_name]['model_flops']
    # cost_100 = ee_stats[next(iter(ee_stats.keys()))]['head_flops'][-1]
    # collect
    ee_results = torch.zeros(len(ee_stats), len(cost_fractions))
    ee_stds = torch.zeros_like(ee_results)
    for i, run_name in enumerate(ee_stats.keys()):
        for j, cost_fraction in enumerate(cost_fractions):
            for k, threshold_cost in enumerate(ee_stats[run_name]['threshold_flops']):
                if threshold_cost > cost_fraction * cost_100:
                    k -= 1
                    break
            ee_results[i, j] = ee_stats[run_name]['threshold_scores'][k].item()
            ee_stds[i, j] = ee_stats[run_name]['threshold_scores_std'][k].item()
    # print
    for i, run_name in enumerate(ee_stats.keys()):
        name = name_dict[run_name]
        print(f'{name}', file=f, end='')
        for j, cost_fraction in enumerate(cost_fractions):
            print(f' & {ee_results[i, j]:.3} \\pm {ee_stds[i, j]:.3}', file=f, end='')
        print(' \\\\\n', file=f, end='')


def main():
    args = parser.parse_args()
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    core_stats = {}
    ee_stats = {}
    point_stats = {}
    name_dict = {}
    display_names = args.exp_names if args.display_names is None else args.display_names
    assert len(args.exp_names) == len(display_names)
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            logging.info(f'Processing for: {run_name} ({args.mode})')
            # TODO possibly split this into two scripts: one that generates outputs for both datasets and one that plots the data
            if args.mode == 'acc':
                state = retrieve_state(args, run_name)
                del state['model_state'], state['optimizer_state']
                if 'thresholds' in state:
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    point_stats[run_name] = state
                else:
                    core_stats[run_name] = state
            elif args.mode == 'calibration':
                model, state = load_model(args, run_name)
                del state['model_state'], state['optimizer_state']
                _, _, id_data = DATASETS_NAME_MAP[state['args'].dataset]()
                id_dataloader = get_loader(id_data, state['args'].batch_size)
                if 'thresholds' in state:
                    id_preds, id_labels = get_preds_earlyexiting(model, id_dataloader)
                    state.update(
                        evaluate_earlyexiting_calibration(id_preds, id_labels, state['head_flops'],
                                                          state['thresholds']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, id_labels = get_preds(model, id_dataloader)
                    state.update(evaluate_calibration(id_preds, id_labels))
                    core_stats[run_name] = state
            elif args.mode == 'ood_detection':
                model, state = load_model(args, run_name)
                del state['model_state'], state['optimizer_state']
                _, _, id_data = DATASETS_NAME_MAP[state['args'].dataset]()
                id_dataloader = get_loader(id_data, state['args'].batch_size)
                _, _, ood_data = DATASETS_NAME_MAP[args.ood_dataset]()
                ood_dataloader = get_loader(ood_data, state['args'].batch_size)
                if 'thresholds' in state:
                    id_preds, _ = get_preds_earlyexiting(model, id_dataloader)
                    ood_preds, _ = get_preds_earlyexiting(model, ood_dataloader)
                    state.update(
                        evaluate_earlyexiting_ood_detection(id_preds, ood_preds, state['head_flops'],
                                                            state['thresholds']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, _ = get_preds(model, id_dataloader)
                    ood_preds, _ = get_preds(model, ood_dataloader)
                    state.update(evaluate_ood_detection(id_preds, ood_preds))
                    core_stats[run_name] = state
    core_stats, point_stats, ee_stats = compute_means_and_stds(args.exp_names, args.exp_ids, core_stats, point_stats,
                                                               ee_stats)
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'{args.output_name}_{args.mode}.txt'
    with open(save_path, 'w') as f:
        generate_score_eff_table(f, core_stats, point_stats, ee_stats, name_dict, PRETTY_NAME_DICT[args.mode])
        logging.info(f'Table saved to {str(save_path)}')


if __name__ == "__main__":
    main()

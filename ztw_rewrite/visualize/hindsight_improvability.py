import argparse
import logging
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import numpy as np
import seaborn as seaborn
import torch
from matplotlib import pyplot as plt

from datasets import DATASETS_NAME_MAP
from eval import get_preds_earlyexiting
from utils import load_model, get_loader
from visualize import mean_std
from visualize.cost_vs_plot import FONT_SIZE, COLORS

parser = argparse.ArgumentParser()
parser.add_argument('--runs_dir', type=Path, default=Path.cwd() / 'runs',
                    help='Root dir where experiment data was saved.')
parser.add_argument('--exp_names', type=str, required=True, nargs='+',
                    help='Unique experiment names to visualize the results for (excluding exp_id).')
parser.add_argument('--exp_ids', type=int, nargs='+', default=[0],
                    help='Experiment ids.')
parser.add_argument('--display_names', type=str, default=None, nargs='+',
                    help='Pretty display names that will be used when generating the plot.')
parser.add_argument('--use_wandb', action='store_true',
                    help='Use W&B. Will save and load the models from the W&B cloud.')
parser.add_argument('--output_dir', type=Path, default=Path.cwd() / 'figures',
                    help='Target directory.')
parser.add_argument('--output_name', type=str, default='hi',
                    help='Output file name prefix to use.')


def get_predictions(args, run_name):
    ee_model, ee_state = load_model(args, run_name)
    if 'thresholds' not in ee_state:
        return None, None
    del ee_state['model_state'], ee_state['optimizer_state']
    _, _, data = DATASETS_NAME_MAP[ee_state['args'].dataset]()
    dataloader = get_loader(data, ee_state['args'].batch_size, shuffle=False)
    ee_preds, labels = get_preds_earlyexiting(ee_model, dataloader)
    return ee_preds, labels


def calculate_hindsight_improvability(ee_preds, labels):
    num_heads = len(ee_preds)
    improvabilities = torch.zeros(num_heads, dtype=torch.float)
    classified_correctly_by_prev = ee_preds[0].argmax(dim=-1) == labels
    for head_idx in range(num_heads):
        current_head_correct = ee_preds[head_idx].argmax(dim=-1) == labels
        current_head_incorrect = ~current_head_correct
        improvabilities[head_idx] = (current_head_incorrect & classified_correctly_by_prev).float().sum() \
                                    / current_head_incorrect.float().sum()
        classified_correctly_by_prev |= current_head_correct
    return improvabilities.tolist()


def calculate_mean_std(exp_names: List[str],
                       exp_ids: List[int],
                       hi_stats: Dict):
    processed_hi_stats = {}
    mean_std(exp_names, exp_ids, hi_stats, processed_hi_stats, None, 'hi')
    return processed_hi_stats


def plot_hindsight_improvability(hi_results: Dict,
                                 name_dict: Dict,
                                 title: str = None):
    seaborn.set_style('whitegrid')
    current_palette = cycle(COLORS)
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    for exp_name, stats in hi_results.items():
        dislay_name = name_dict[exp_name]
        if 'hi' not in stats:
            continue
        color = next(current_palette)
        improvabilities = stats['hi'].numpy()
        xs = np.arange(len(improvabilities)) + 1
        ax.plot(xs, improvabilities, color=color)
        plt.scatter(xs, improvabilities, s=60, label=dislay_name, color=color)
        if 'hi_std' in stats:
            stds = stats['hi_std'].numpy()
            ax.errorbar(xs, improvabilities, xerr=np.zeros_like(xs), yerr=stds, ecolor=color, fmt=' ', alpha=0.5)
    # ax.legend(loc='upper left', prop={'size': FONT_SIZE - 4})
    ax.legend(loc='best', prop={'size': FONT_SIZE - 4})
    ax.set_title(title, fontdict={'fontsize': FONT_SIZE + 1})
    ax.set_xlabel('IC', fontsize=FONT_SIZE)
    ax.set_ylabel('Hindsight Improvability', fontsize=FONT_SIZE)
    # ax.set_xlim(right=1.1 * baseline_ops)
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(baseline_ops / 4))
    # ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=baseline_ops))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    fig.set_tight_layout(True)
    return fig


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
    hi_stats = {}
    name_dict = {}
    display_names = args.display_names if args.display_names is not None else args.run_names
    assert len(args.exp_names) == len(display_names)
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            logging.info(f'Calculating hindsight improvabilities for: {run_name}')
            ee_preds, labels = get_predictions(args, run_name)
            if ee_preds is not None:
                stats = {'hi': calculate_hindsight_improvability(ee_preds, labels)}
                hi_stats[run_name] = stats
    hi_stats = calculate_mean_std(args.exp_names, args.exp_ids, hi_stats)
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    fig = plot_hindsight_improvability(hi_stats, name_dict)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'{args.output_name}.png'
    fig.savefig(save_path)
    logging.info(f'Figure saved in {str(save_path)}')
    plt.close(fig)


if __name__ == "__main__":
    main()

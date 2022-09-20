import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from datasets import DATASETS_NAME_MAP
from eval import get_preds_earlyexiting, get_preds
from utils import load_model, get_loader
from visualize.cost_vs_plot import FONT_SIZE

parser = argparse.ArgumentParser()
parser.add_argument('--runs_dir', type=Path, default=Path.cwd() / 'runs',
                    help='Root dir where experiment data was saved.')
parser.add_argument('--run_names', type=str, required=True, nargs='+',
                    help='Unique run names to visualize the results for.')
parser.add_argument('--display_names', type=str, default=None, nargs='+',
                    help='Pretty display names that will be used when generating the plot.')
parser.add_argument('--reference_run_name', type=str, default=None,
                    help='Unique run name for plotting the other model reference diversity.')
parser.add_argument('--use_wandb', action='store_true',
                    help='Use W&B. Will save and load the models from the W&B cloud.')
parser.add_argument('--output_dir', type=Path, default=Path.cwd() / 'figures',
                    help='Target directory.')
parser.add_argument('--output_name', type=str, default='diversity',
                    help='Output file name prefix to use.')
parser.add_argument('--ood_dataset', type=str, default=None,
                    help='Out-of-Distribution dataset to use instead of the in-Distribution dataset.')


def count_diversity_for_incorrectly_classified(preds_left, preds_right, labels):
    if labels is not None:
        incorrectly_left = preds_left.argmax(dim=-1) != labels
        incorrectly_right = preds_right.argmax(dim=-1) != labels
        preds_mask = incorrectly_left & incorrectly_right
        preds_left = preds_left[preds_mask]
        preds_right = preds_right[preds_mask]
    assert preds_left.size(0) == preds_right.size(0)
    answer_left = preds_left.argmax(dim=-1)
    answer_right = preds_right.argmax(dim=-1)
    return (answer_left != answer_right).sum() / preds_left.size(0)


def head_diversity_matrix(preds, labels):
    num_heads = len(preds)
    diversity_matrix = torch.zeros(num_heads, num_heads, dtype=torch.float)
    for i in range(num_heads):
        for j in range(num_heads):
            diversity_matrix[i, j] = count_diversity_for_incorrectly_classified(preds[i], preds[j], labels)
    return diversity_matrix


def reference_diversity_row(ref_preds, head_preds, labels):
    num_heads = len(head_preds)
    diversity_row = torch.zeros(num_heads, dtype=torch.float)
    for i in range(num_heads):
        diversity_row[i] = count_diversity_for_incorrectly_classified(ref_preds, head_preds[i], labels)
    return diversity_row


def diversities_to_figure(matrix, reference_row=None, xlabel="", ylabel=""):
    if reference_row is None:
        matrix = matrix.cpu().numpy()
    else:
        matrix = torch.cat([matrix, reference_row.unsqueeze(0)]).cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    ax.imshow(1.0 - matrix, cmap='plasma', vmin=0, vmax=1)
    # set x axis
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(i) for i in np.arange(matrix.shape[1])], fontsize=18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    # set y axis
    ax.set_yticks(np.arange(matrix.shape[0]))
    if reference_row is None:
        ax.set_yticklabels([str(i) for i in np.arange(matrix.shape[0])], fontsize=18)
    else:
        ax.set_yticklabels([str(i) for i in np.arange(matrix.shape[0] - 1)] + ['R'], fontsize=18)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    # plot text
    for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):
        ax.text(j, i, f'{matrix[i, j]:4.2f}' if matrix[i, j] != 0 else '.', horizontalalignment='center', fontsize=14,
                verticalalignment='center', color='black')
    ax.autoscale()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    fig.set_tight_layout(True)
    return fig


def get_predictions(args, run_name):
    ee_model, ee_state = load_model(args, run_name)
    del ee_state['model_state'], ee_state['optimizer_state']
    if args.ood_dataset is None:
        _, _, id_data = DATASETS_NAME_MAP[ee_state['args'].dataset]()
        dataloader = get_loader(id_data, ee_state['args'].batch_size, shuffle=False)
    else:
        _, _, ood_data = DATASETS_NAME_MAP[args.ood_dataset]()
        dataloader = get_loader(ood_data, ee_state['args'].batch_size, shuffle=False)
    ee_preds, labels = get_preds_earlyexiting(ee_model, dataloader)
    if args.reference_run_name is not None:
        ref_model, ref_state = load_model(args, args.reference_run_name)
        del ref_state
        ref_preds, ref_labels = get_preds(ref_model, dataloader)
        assert torch.all(labels == ref_labels)
    else:
        ref_preds = None
    if args.ood_dataset is not None:
        labels = None
    return ee_preds, ref_preds, labels


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # TODO add seeds / standard deviations support?
    display_names = args.display_names if args.display_names is not None else args.run_names
    for run_name, display_name in zip(args.run_names, display_names):
        logging.info(f'Calculating head diversities for: {run_name}')
        ee_preds, ref_preds, labels = get_predictions(args, run_name)
        diversity_matrix = head_diversity_matrix(ee_preds, labels)
        reference_diversity = reference_diversity_row(ref_preds, ee_preds, labels) if ref_preds is not None else None
        fig = diversities_to_figure(diversity_matrix, reference_diversity)
        save_path_postfix = '' if args.ood_dataset is None else f'_{args.ood_dataset}'
        save_path = args.output_dir / f'{args.output_name}_{display_name}{save_path_postfix}.png'
        fig.savefig(save_path)
        logging.info(f'Figure saved in {str(save_path)}')
        plt.close(fig)


if __name__ == "__main__":
    main()

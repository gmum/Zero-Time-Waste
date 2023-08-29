import logging
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import matplotlib
import seaborn as seaborn
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from datasets import DATASETS_NAME_MAP
from eval import evaluate_earlyexiting_calibration, evaluate_earlyexiting_ood_detection, \
    evaluate_calibration, evaluate_ood_detection, get_preds_earlyexiting, get_preds
from utils import retrieve_final, load_model, get_loader
from visualize import mean_std


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.output_name = 'cost_vs'  # Output file name prefix to use.
    default_args.mode = 'acc'  # Type of plot to generate. Choices: ['acc', 'calibration', 'ood_detection']
    default_args.ood_dataset = None  # Out-of-Distribution dataset.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    return default_args


PRETTY_NAME_DICT = {
    'acc': 'Accuracy',
    'calibration': 'Calibration Error',
    'ood_detection': 'OOD detection AUROC',
}

FONT_SIZE = 22
COLORS = ['#000000', '#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#810f7c', '#0ee5a8', '#f04f6d', '#00ffff', '#0000ff']


def mark_orig_result(stats: Dict, ax: matplotlib.axes.SubplotBase, name: str, color: str):
    cost = stats['model_flops'].numpy()
    score_mean = stats['final_score'].numpy()
    ax.scatter(cost, score_mean, marker='X', label=name, color=color, s=250, zorder=3, linewidths=0.)
    if 'final_score_std' in stats:
        x_std = stats['model_flops_std'].numpy()
        score_std = stats['final_score_std'].numpy()
        ax.errorbar(cost, score_mean, xerr=x_std, yerr=score_std, ecolor=color, alpha=0.5)


def draw_for_points(stats: Dict, ax: matplotlib.axes.SubplotBase, name: str, color: str):
    costs = stats['final_flops'].numpy()
    scores = stats['final_scores'].numpy()
    ax.scatter(costs, scores, marker='X', label=f'{name}', color=color, s=200, zorder=3, linewidths=0.)
    if 'final_scores_std' in stats:
        x_std = stats['final_flops_std'].numpy()
        score_stds = stats['final_scores_std'].numpy()
        ax.errorbar(costs, scores, xerr=x_std, yerr=score_stds, ecolor=color, alpha=0.5)


def draw_for_ics(stats: Dict, ax: matplotlib.axes.SubplotBase, name: str, color: str):
    costs = stats['head_flops'].numpy()
    scores = stats['head_scores'].numpy()
    # ax.scatter(costs, scores, marker='X', label=f'{name} IC', color=color, s=125, zorder=3, edgecolors='black', linewidths=1)
    # ax.scatter(costs, scores, marker='X', color=color, s=125, zorder=3, edgecolors='black', linewidths=1)
    ax.scatter(costs, scores, marker='o', color=color, s=90, zorder=3, edgecolors='black', linewidths=0)
    if 'head_scores_std' in stats:
        x_std = stats['head_flops_std'].numpy()
        head_stds = stats['head_scores_std'].numpy()
        ax.errorbar(costs, scores, xerr=x_std, yerr=head_stds, ecolor=color, fmt=' ', alpha=0.5)


def draw_for_thresholds(stats: Dict,
                        ax: matplotlib.axes.SubplotBase,
                        name: str,
                        color: str):
    # thresholds = stats['thresholds'].numpy()
    costs = stats['threshold_flops'].numpy()
    scores = stats['threshold_scores'].numpy()
    ax.plot(costs, scores, label=name, color=color)
    if 'threshold_scores_std' in stats:
        scores_stds = stats['threshold_scores_std'].numpy()
        ax.fill_between(costs, scores - scores_stds, scores + scores_stds, alpha=0.3, color=color)


def plot_score_eff_tradeoff(core_stats: Dict,
                            point_stats: Dict,
                            ee_stats: Dict,
                            name_dict: Dict[str, str],
                            x_label: str = None,
                            title: str = None):
    seaborn.set_style('whitegrid')
    current_palette = cycle(COLORS)
    colors = {}
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    for run_name, stats in core_stats.items():
        colors[run_name] = next(current_palette)
        mark_orig_result(stats, ax, name_dict[run_name], colors[run_name])
    for run_name, stats in point_stats.items():
        colors[run_name] = next(current_palette)
        draw_for_points(stats, ax, name_dict[run_name], colors[run_name])
    for run_name, stats in ee_stats.items():
        colors[run_name] = next(current_palette)
        draw_for_ics(stats, ax, name_dict[run_name], colors[run_name])
        draw_for_thresholds(stats, ax, name_dict[run_name], colors[run_name])
    # ax.legend(loc='upper left', prop={'size': FONT_SIZE - 4})
    ax.legend(loc='best', prop={'size': FONT_SIZE - 4})
    ax.set_title(title, fontdict={'fontsize': FONT_SIZE + 1})
    ax.set_xlabel('Inference FLOPs', fontsize=FONT_SIZE)
    # ax.set_xlabel('Inference Time', fontsize=FONT_SIZE)
    ax.set_ylabel(x_label, fontsize=FONT_SIZE)
    # ax.set_xlim(right=1.1 * baseline_ops)
    assert len(core_stats) > 0 or len(point_stats) > 0 or len(ee_stats) > 0
    # base_model_run_name = next(iter(core_stats.keys()))
    # baseline_ops = core_stats[base_model_run_name]['model_flops'].numpy()
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(baseline_ops / 4))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=baseline_ops))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FONT_SIZE - 4)
    # labels - add pretty "ICs"
    handles, labels = ax.get_legend_handles_labels()
    circles = []
    # for k, color in colors.items():
    #     if k == base_model_run_name:
    #         continue
    #     circles += [matplotlib.lines.Line2D([], [], color=color, marker='o', linestyle='None',
    #                                         markersize=10, label='IC')]
    # if len(ee_stats) > 0:
    #     handles += [tuple(circles)]
    #     labels += ["IC"]
    #     if 'Base' in labels:
    #         index = labels.index('Base')
    #         labels[index] = 'Base Network'
    #     logging.info(f'handles: {handles}')
    #     logging.info(f'labels: {labels}')
    #     ax.legend(handles=handles, labels=labels, prop={'size': FONT_SIZE},
    #               handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.set_tight_layout(True)
    return fig


def compute_means_and_stds(exp_names: List[str],
                           exp_ids: List[int],
                           core_stats: Dict,
                           point_stats: Dict,
                           ee_stats: Dict):
    processed_core_stats = {}
    processed_point_stats = {}
    processed_ee_stats = {}
    mean_std(exp_names, exp_ids, core_stats, processed_core_stats, 'model_flops', 'final_score')
    mean_std(exp_names, exp_ids, point_stats, processed_point_stats, 'final_flops', 'final_scores')
    mean_std(exp_names, exp_ids, ee_stats, processed_ee_stats, 'head_flops', 'head_scores')
    mean_std(exp_names, exp_ids, ee_stats, processed_ee_stats, 'threshold_flops', 'threshold_scores')
    # copy_entry(exp_names, exp_ids, ee_stats, processed_ee_stats, 'thresholds')
    return processed_core_stats, processed_point_stats, processed_ee_stats


def main(args):
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
    accelerator = Accelerator(split_batches=True)
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            logging.info(f'Processing for: {run_name} ({args.mode})')
            # TODO possibly split this into two scripts: one that generates outputs for both datasets and one that plots the data
            if args.mode == 'acc':
                final_results = retrieve_final(args, run_name)
                del final_results['model_state']
                if 'thresholds' in final_results:
                    ee_stats[run_name] = final_results
                elif 'hyperparam_values' in final_results:
                    point_stats[run_name] = final_results
                elif 'final_score' in final_results:
                    core_stats[run_name] = final_results
                else:
                    logging.info(f'Skipping {run_name} as it has not any recognizable data to plot')
            elif args.mode == 'calibration':
                model, run_args, final_results = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                del final_results['model_state']
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](run_args.dataset_args)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                if 'thresholds' in final_results:
                    id_preds, id_labels = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    final_results.update(
                        evaluate_earlyexiting_calibration(id_preds, id_labels,
                                                          final_results['head_flops'],
                                                          final_results['thresholds']))
                    ee_stats[run_name] = final_results
                elif 'hyperparam_values' in final_results:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, id_labels = get_preds(accelerator, model, id_dataloader)
                    final_results.update(evaluate_calibration(id_preds, id_labels))
                    core_stats[run_name] = final_results
            elif args.mode == 'ood_detection':
                model, run_args, final_results = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                del final_results['model_state']
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](run_args.dataset_args)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                _, _, ood_data = DATASETS_NAME_MAP[args.ood_dataset](run_args.dataset_args)
                ood_dataloader = get_loader(ood_data, run_args.batch_size, accelerator)
                if 'thresholds' in final_results:
                    id_preds, _ = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds_earlyexiting(accelerator, model, ood_dataloader)
                    final_results.update(
                        evaluate_earlyexiting_ood_detection(id_preds, ood_preds, final_results['head_flops'],
                                                            final_results['thresholds']))
                    ee_stats[run_name] = final_results
                elif 'hyperparam_values' in final_results:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, _ = get_preds(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds(accelerator, model, ood_dataloader)
                    final_results.update(evaluate_ood_detection(id_preds, ood_preds))
                    core_stats[run_name] = final_results
    core_stats, point_stats, ee_stats = compute_means_and_stds(args.exp_names, args.exp_ids, core_stats, point_stats,
                                                               ee_stats)
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    fig = plot_score_eff_tradeoff(core_stats, point_stats, ee_stats, name_dict, PRETTY_NAME_DICT[args.mode])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'{args.output_name}_{args.mode}.png'
    fig.savefig(save_path)
    logging.info(f'Figure saved in {str(save_path)}')
    plt.close(fig)


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)

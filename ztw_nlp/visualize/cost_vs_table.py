import logging
import math

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from datasets_config import DATASETS_NAME_MAP
from eval import evaluate_earlyexiting_calibration, evaluate_earlyexiting_ood_detection, \
    evaluate_calibration, evaluate_ood_detection, get_preds_earlyexiting, get_preds
from utils import retrieve_final, load_model, get_loader
from visualize.cost_vs_plot import PRETTY_NAME_DICT, compute_means_and_stds, get_default_args


def generate_score_eff_table(f, core_stats, point_stats, ee_stats, name_dict, param):
    cost_fractions = [0.25, 0.5, 0.75, 1.0, math.inf]
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
                    if k > 0:
                        result_valid = True
                        k -= 1
                    else:
                        result_valid = False
                    break
            if result_valid:
                ee_results[i, j] = ee_stats[run_name]['threshold_scores'][k].item()
                ee_stds[i, j] = ee_stats[run_name]['threshold_scores_std'][k].item()
            else:
                ee_results[i, j] = math.nan
                ee_stds[i, j] = math.nan
    # print
    for i, run_name in enumerate(ee_stats.keys()):
        name = name_dict[run_name]
        print(f'{name}', file=f, end='')
        for j in range(ee_results.size(-1)):
            result = f'${ee_results[i, j] * 100:.2f}$' if not math.isnan(ee_results[i, j]) else '-'
            result_std = f' $ \\pm {ee_stds[i, j] * 100:.2f}$' if not math.isnan(ee_stds[i, j]) else ''
            print(f' & {result}{result_std}', file=f, end='')
        print(' \\\\\n', file=f, end='')


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
                state = retrieve_final(args, run_name)
                if 'thresholds' in state:
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    point_stats[run_name] = state
                else:
                    core_stats[run_name] = state
            elif args.mode == 'calibration':
                model, run_args, state = load_model(args, exp_name, exp_id)
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](run_args.dataset_args)
                model = accelerator.prepare(model)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                if 'thresholds' in state:
                    id_preds, id_labels = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    state.update(
                        evaluate_earlyexiting_calibration(id_preds, id_labels, state['head_flops'],
                                                          state['thresholds']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, id_labels = get_preds(accelerator, model, id_dataloader)
                    state.update(evaluate_calibration(id_preds, id_labels))
                    core_stats[run_name] = state
            elif args.mode == 'ood_detection':
                model, run_args, state = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](run_args.dataset_args)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                _, _, ood_data = DATASETS_NAME_MAP[args.ood_dataset](run_args.dataset_args)
                ood_dataloader = get_loader(ood_data, run_args.batch_size, accelerator)
                if 'thresholds' in state:
                    id_preds, _ = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds_earlyexiting(accelerator, model, ood_dataloader)
                    state.update(
                        evaluate_earlyexiting_ood_detection(id_preds, ood_preds, state['head_flops'],
                                                            state['thresholds']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, _ = get_preds(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds(accelerator, model, ood_dataloader)
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
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
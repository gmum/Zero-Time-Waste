import torch


def mean_std(exp_names, exp_ids, in_stats, out_stats, x_key_name, y_key_name):
    x_std_key_name = f'{x_key_name}_std'
    y_std_key_name = f'{y_key_name}_std'
    for exp_name in exp_names:
        x_scores, y_scores = [], []
        for exp_id in exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            if run_name not in in_stats:
                continue
            if x_key_name is not None:
                assert x_key_name in set(in_stats[run_name]), f'{x_key_name} not in stats of run {run_name}'
                x_scores.append(in_stats[run_name][x_key_name])
            if y_key_name is not None:
                assert y_key_name in set(in_stats[run_name]), f'{y_key_name} not in stats of run {run_name}'
                y_scores.append(in_stats[run_name][y_key_name])
        if (len(y_scores) == 0 or len(x_scores) == 0) and (x_key_name is not None and y_key_name is not None):
            continue
        if exp_name not in out_stats:
            out_stats[exp_name] = {}
        if x_key_name is not None and len(x_scores) > 0:
            x_scores = torch.tensor(x_scores)
            x_std, x_mean = torch.std_mean(x_scores.double(), dim=0, unbiased=True)
            out_stats[exp_name][x_key_name] = x_mean
            out_stats[exp_name][x_std_key_name] = x_std
        if y_key_name is not None and len(y_scores) > 0:
            y_scores = torch.tensor(y_scores)
            y_std, y_mean = torch.std_mean(y_scores, dim=0, unbiased=True)
            out_stats[exp_name][y_key_name] = y_mean
            out_stats[exp_name][y_std_key_name] = y_std


def copy_entry(exp_names, exp_ids, in_stats, out_stats, key_name):
    for exp_name in exp_names:
        for exp_id in exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            if run_name not in in_stats:
                continue
            else:
                out_stats[exp_name][key_name] = in_stats[run_name][key_name]
                break

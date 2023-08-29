import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.early_exit import train as train_ee
from methods.l2w import train as train_l2w
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.cost_vs_table import main as cost_vs_table


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'ee_eval'

    account = ''
    # account = None
    qos = 'normal'
    partition = ''
    # partition = 'dgxmatinf,rtx3080'
    # partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2

    gpus_per_task = 1
    cpus_per_gpu = 16
    mem_per_gpu = '64G'

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_gpus_per_task=gpus_per_task,
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_gpu=mem_per_gpu,
    )

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()
    exp_ids = [1, 2, 3]
    # exp_ids = [1]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    common_args.dataset = 'imagenet'
    common_args.dataset_args = {}
    common_args.dataset_args.variant = 'tv_efficientnet_v2_s'
    common_args.use_wandb = True
    common_args.mixup_alpha = None
    common_args.cutmix_alpha = None
    common_args.batch_size = 64
    common_args.epochs = 15
    common_args.eval_points = 20
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.001
    common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'cosine'
    common_args.scheduler_args = {}

    # ═══════════════════════════════════════════════════════════════════ #

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = 'tv_efficientnet_s'
    base_model_args.model_args = {}
    base_model_args.epochs = 0  # pretrained
    base_model_args.eval_points = 0

    # ════════════════════════ train base models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'EfficientNet S')
    base_exp_name = exp_name

    # ════════════════════════ SDN model settings ════════════════════════ #

    sdn_model_args = deepcopy(common_args)
    sdn_model_args.with_backbone = False
    sdn_model_args.model_class = 'sdn'
    sdn_model_args.model_args = {}
    sdn_model_args.model_args.head_type = 'conv'
    # ~40 layers
    sdn_model_args.model_args.place_at = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    sdn_model_args.optimizer_class = 'adam'
    sdn_model_args.optimizer_args = {}
    sdn_model_args.optimizer_args.lr = 0.0005
    sdn_model_args.optimizer_args.weight_decay = 0.1

    # ════════════════════════ train SDN models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(sdn_model_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name
        base_run_name = f'{base_exp_name}_{exp_id}'
        dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
        executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
        job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'SDN')
    sdn_exp_name = exp_name

    # ════════════════════════ PBEE model settings ════════════════════════ #

    pbee_model_args = deepcopy(common_args)
    pbee_model_args.with_backbone = False
    pbee_model_args.model_class = 'pbee'
    pbee_model_args.model_args = {}
    pbee_model_args.optimizer_class = 'adam'
    pbee_model_args.epochs = 0

    # ════════════════════════ train PBEE models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(pbee_model_args)
        args.exp_id = exp_id
        args.base_on = sdn_exp_name
        base_run_name = f'{sdn_exp_name}_{exp_id}'
        dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
        executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
        job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'PBEE')

    # ════════════════════════ GPF model settings ════════════════════════ #

    gpf_model_args = deepcopy(common_args)
    gpf_model_args.with_backbone = False
    gpf_model_args.model_class = 'gpf'
    gpf_model_args.model_args = {}
    gpf_model_args.model_args.head_type = 'conv'
    gpf_model_args.model_args.place_at = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    gpf_model_args.model_args.head_dim = 192
    gpf_model_args.model_args.state_dropout = 0.0
    gpf_model_args.optimizer_class = 'adam'
    gpf_model_args.optimizer_args = {}
    gpf_model_args.optimizer_args.lr = 0.0005
    gpf_model_args.optimizer_args.weight_decay = 0.0

    # ════════════════════════ train GPF models ════════════════════════ #

    for lr, wd in [
        # (0.0005, 0.0),
        # (0.0001, 0.0),
        (0.0001, 0.1),
        # (0.0001, 0.01),
        # (0.0001, 0.001),
        # (0.0001, 0.0001),
        # (0.0002, 0.0),
        # (0.0003, 0.0),
        # (0.0004, 0.0),
        # (0.00005, 0.0),
        # (0.00001, 0.0),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(gpf_model_args)
            args.exp_id = exp_id
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.base_on = base_exp_name
            base_run_name = f'{base_exp_name}_{exp_id}'
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
            job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
            jobs.append(job)
            exp_name, run_name = generate_run_name(args)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(f'GPF')
        # display_names.append(f'GPF lr={lr} wd={wd}')

    # ════════════════════════ L2W model settings ════════════════════════ #

    l2w_model_args = deepcopy(common_args)
    l2w_model_args.with_backbone = False
    l2w_model_args.model_class = 'l2w'
    l2w_model_args.model_args = {}
    l2w_model_args.model_args.head_type = 'conv'
    l2w_model_args.model_args.place_at = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    l2w_model_args.optimizer_class = 'sgd'
    l2w_model_args.optimizer_args = {}
    l2w_model_args.optimizer_args.lr = 0.0005
    l2w_model_args.optimizer_args.momentum = 0.9
    l2w_model_args.optimizer_args.weight_decay = 0.1
    l2w_model_args.l2w_meta_interval = 100
    l2w_model_args.wpn_width = 500
    l2w_model_args.wpn_depth = 1
    l2w_model_args.wpn_optimizer_class = 'adam'
    l2w_model_args.wpn_optimizer_args = {}
    l2w_model_args.wpn_optimizer_args.lr = 1e-4
    l2w_model_args.wpn_optimizer_args.weight_decay = 1e-4
    l2w_model_args.wpn_scheduler_class = 'cosine'
    l2w_model_args.wpn_scheduler_args = {}
    l2w_model_args.l2w_epsilon = 0.3
    l2w_model_args.l2w_target_p = 15
    l2w_model_args.clip_grad_norm = 10.0
    #
    l2w_model_args.batch_size = 32

    # ════════════════════════ train L2W models ════════════════════════ #

    for lr, wd in [
        # (20.0, 0.0),
        # (10.0, 0.0),
        # (1.0, 0.0),
        # (0.1, 0.0),
        # (0.01, 0.1),
        # (0.01, 0.01),
        # (0.01, 0.001),
        (0.01, 0.0001),
        # (0.01, 0.0),
        # (0.001, 0.0),
        # (0.0001, 0.0),
        # (0.00001, 0.0),
        # (0.001, 0.1),
        # (0.001, 0.01),
        # (0.001, 0.001),
        # (0.001, 0.0001),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(l2w_model_args)
            args.exp_id = exp_id
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.base_on = base_exp_name
            base_run_name = f'{base_exp_name}_{exp_id}'
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
            job = submit_job(executor, train_l2w, args, num_gpus=gpus_per_task)
            jobs.append(job)
            exp_name, run_name = generate_run_name(args)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(f'L2W')
        # display_names.append(f'L2W lr={lr} wd={wd}')

    # ════════════════════════ ZTW cascading model settings ════════════════════════ #

    cascading_model_args = deepcopy(common_args)
    cascading_model_args.with_backbone = False
    cascading_model_args.model_class = 'ztw_cascading'
    cascading_model_args.model_args = {}
    cascading_model_args.model_args.head_type = 'conv_cascading'
    cascading_model_args.model_args.place_at = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                                                38]
    cascading_model_args.optimizer_class = 'adam'
    cascading_model_args.optimizer_args = {}
    cascading_model_args.optimizer_args.lr = 0.0005
    cascading_model_args.optimizer_args.weight_decay = 0.1

    # ════════════════════════ train ZTW cascading models ════════════════════════ #

    for lr, wd in [
        (0.0005, 0.1),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(cascading_model_args)
            args.exp_id = exp_id
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.base_on = base_exp_name
            base_run_name = f'{base_exp_name}_{exp_id}'
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
            job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
            jobs.append(job)
            exp_name, run_name = generate_run_name(args)
            run_to_job_map[run_name] = job
        # exp_names.append(exp_name)
        # display_names.append(f'Cascading')
        cascading_exp_name = exp_name

    # ════════════════════════ ZTW ensembling model settings ════════════════════════ #

    ensembling_model_args = deepcopy(common_args)
    ensembling_model_args.with_backbone = False
    ensembling_model_args.model_class = 'ztw_ensembling'
    ensembling_model_args.model_args = {}
    ensembling_model_args.epochs = 2
    ensembling_model_args.batch_size = 64
    ensembling_model_args.optimizer_class = 'adam'
    ensembling_model_args.optimizer_args = {}
    ensembling_model_args.optimizer_args.lr = 0.0005
    ensembling_model_args.optimizer_args.weight_decay = 0.0

    # ════════════════════════ train ZTW ensembling models ════════════════════════ #

    for optimizer, lr, wd in [
        # ('adam', 0.1, 0.0),
        # ('adam', 0.01, 0.0),
        ('adam', 0.001, 0.0),
        # ('adam', 0.0001, 0.0),
        # ('adam', 0.00001, 0.0),
        #
        # ('sgd', 10.0, 0.0),
        # ('sgd', 1.0, 0.0),
        # ('sgd', 0.1, 0.0),
        # ('sgd', 0.01, 0.0),
        # ('sgd', 0.001, 0.0),
        # ('sgd', 0.0001, 0.0),
        # ('sgd', 0.00001, 0.0),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(ensembling_model_args)
            args.exp_id = exp_id
            args.optimizer_class = optimizer
            args.optimizer_args = {}
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.base_on = cascading_exp_name
            base_run_name = f'{cascading_exp_name}_{exp_id}'
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
            job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
            jobs.append(job)
            exp_name, run_name = generate_run_name(args)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        # display_names.append(f'ZTW o={optimizer} lr={lr} wd={wd}')
        display_names.append(f'ZTW')

    # ═════════════════════════════════════════════════════════ #

    print(f'Exp names: {exp_names}')
    print(f'Display names: {display_names}')
    print(f'SLURM JIDs: {[job.job_id for job in jobs]}')

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()
    out_dir_name = f'ztw_efficientnet_v2_{common_args.dataset}'
    output_dir = Path(os.environ['RESULTS_DIR']) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = 'cost_vs'
    plot_args.mode = 'acc'
    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(stderr_to_stdout=True,
                               timeout_min=timeout,
                               slurm_job_name=job_name,
                               slurm_account=account,
                               slurm_qos=qos,
                               slurm_partition=partition,
                               slurm_ntasks_per_node=1,
                               slurm_gpus_per_task=1,
                               slurm_cpus_per_gpu=cpus_per_gpu,
                               slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={'dependency': dependency_str})
    submit_job(executor, cost_vs_plot, plot_args)

    # ════════════════════════ generate a table ════════════════════════ #

    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(stderr_to_stdout=True,
                               timeout_min=timeout,
                               slurm_job_name=job_name,
                               slurm_account=account,
                               slurm_qos=qos,
                               slurm_partition=partition,
                               slurm_ntasks_per_node=1,
                               slurm_gpus_per_task=1,
                               slurm_cpus_per_gpu=cpus_per_gpu,
                               slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={'dependency': dependency_str})
    submit_job(executor, cost_vs_table, plot_args)

    # ════════════════════════ plot cost vs calibration ════════════════════════ #
    # plot only for backbone, SDN and ZTW for readability
    plot_args.exp_names = exp_names[:2] + exp_names[-1:]
    plot_args.display_names = display_names[:2] + display_names[-1:]
    plot_args.mode = 'calibration'
    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(stderr_to_stdout=True,
                               timeout_min=timeout,
                               slurm_job_name=job_name,
                               slurm_account=account,
                               slurm_qos=qos,
                               slurm_partition=partition,
                               slurm_ntasks_per_node=1,
                               slurm_gpus_per_task=1,
                               slurm_cpus_per_gpu=cpus_per_gpu,
                               slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={'dependency': dependency_str})
    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == '__main__':
    main()

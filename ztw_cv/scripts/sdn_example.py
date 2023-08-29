import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.early_exit import train as train_ee
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot


# command:
# python -m scripts.sdn_example


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "ee_eval"

    account = ""
    # account = None

    qos = "normal"

    partition = ""
    # partition = 'dgxmatinf,rtx3080'
    # partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2

    gpus_per_task = 1
    cpus_per_gpu = 16
    mem_per_gpu = "64G"

    executor = submitit.AutoExecutor(folder=os.environ["LOGS_DIR"])
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

    # exp_ids = [1, 2, 3]
    exp_ids = [1]

    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.dataset = "tinyimagenet"
    common_args.use_wandb = True
    common_args.mixup_alpha = 0.0
    common_args.batch_size = 128
    common_args.epochs = 100
    common_args.eval_points = 20

    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0

    common_args.optimizer_class = "adam"
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.01
    common_args.optimizer_args.weight_decay = 0.0005

    common_args.scheduler_class = "cosine"
    common_args.scheduler_args = {}

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)

    base_model_args.model_class = "resnet50"
    base_model_args.model_args = {}
    base_model_args.model_args.num_classes = 200

    # ════════════════════════ SDN model settings ════════════════════════ #

    sdn_model_args = deepcopy(common_args)

    sdn_model_args.batch_size = 128
    sdn_model_args.epochs = 50
    sdn_model_args.with_backbone = True

    sdn_model_args.model_class = "sdn"
    sdn_model_args.model_args = {}
    sdn_model_args.model_args.head_type = "standard"
    sdn_model_args.model_args.place_at = [4, 6, 8, 10, 12, 14]

    sdn_model_args.optimizer_args = {}
    sdn_model_args.optimizer_args.lr = 0.001
    sdn_model_args.optimizer_args.weight_decay = 0.00005

    # ════════════════════════ train base models ════════════════════════ #

    jobs = []
    base_jobs = {}
    exp_names = []
    display_names = []

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id

        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        base_jobs[exp_id] = job

    base_model_exp_name = generate_run_name(args)[0]
    exp_names.append(base_model_exp_name)
    display_names.append(f"ResNet-50")

    # ════════════════════════ train SDN models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(sdn_model_args)
        args.exp_id = exp_id
        args.base_on = base_model_exp_name  # path to the base model will be inferred from the experiment name

        # this makes the sdn job wait for the base model to finish
        dependency_str = f"afterany:{base_jobs[exp_id].job_id}"

        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
        jobs.append(job)

    sdn_model_exp_name = generate_run_name(args)[0]
    exp_names.append(sdn_model_exp_name)
    display_names.append(f"SDN")

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()
    out_dir_name = f"ee_{common_args.dataset}"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False
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
                               slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == "__main__":
    main()

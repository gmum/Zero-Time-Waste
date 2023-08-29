import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from datasets_config import MODEL_TO_TOKENIZER_NAME, DATASET_TO_SEQUENCE_LENGTH, DATASET_TO_NUM_CLASSES
from methods.early_exit import train as train_ee
from methods.l2w import train as train_l2w
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.cost_vs_table import main as cost_vs_table

DATASET = "sst2"

MODEL = 'bert_base'
DISPLAY_BASE = "BERT-base"
NUM_ICS = 11
TOKENIZER_NAME = MODEL_TO_TOKENIZER_NAME[MODEL]
NUM_CLASSES = DATASET_TO_NUM_CLASSES[DATASET]
SEQ_LEN = DATASET_TO_SEQUENCE_LENGTH[DATASET]
DS_ARGS = {'tokenizer_name': TOKENIZER_NAME, 'max_seq_length': SEQ_LEN}
MODEL_ARGS = {'num_classes': NUM_CLASSES, 'max_seq_length': SEQ_LEN}

DS_TO_BASE_ARGS = {
    'rte': {'lr': 2e-5, 'batch_size': 16, 'epochs': 5},
    'mrpc': {'lr': 2e-5, 'batch_size': 16, 'epochs': 5},
    'sst2': {'lr': 1e-5, 'batch_size': 16, 'epochs': 3},
    'qqp': {'lr': 5e-5, 'batch_size': 32, 'epochs': 3},
    'qnli': {'lr': 1e-5, 'batch_size': 16, 'epochs': 3},
}
BASE_EPOCHS = DS_TO_BASE_ARGS[DATASET]['epochs']
BASE_LR = DS_TO_BASE_ARGS[DATASET]['lr']
BASE_BATCH_SIZE = DS_TO_BASE_ARGS[DATASET]['batch_size']

EE_BATCH_SIZE = 32


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'ee_eval'

    account = None
    qos = 'normal'
    partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2
    gpus_per_task = 'ampere:1'
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
    # exp_ids = [1, 2, 3]
    exp_ids = [1, 2, 3]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    common_args.dataset = DATASET
    common_args.dataset_args = DS_ARGS
    common_args.use_wandb = True
    common_args.mixup_alpha = None
    common_args.cutmix_alpha = None
    # common_args.batch_size = 64
    # common_args.epochs = 5
    # common_args.eval_points = 20
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    # common_args.optimizer_args.lr = 0.001
    # common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'cosine'
    common_args.scheduler_args = {}

    # ═══════════════════════════════════════════════════════════════════ #

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = MODEL
    base_model_args.model_args = MODEL_ARGS
    base_model_args.epochs = BASE_EPOCHS
    base_model_args.batch_size = BASE_BATCH_SIZE
    base_model_args.eval_points = 20
    base_model_args.optimizer_args.lr = BASE_LR
    base_model_args.optimizer_args.weight_decay = 0.0005
    base_model_args.scheduler_class = "cosine"

    # ════════════════════════ train base models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(DISPLAY_BASE)
    base_exp_name = exp_name

    # ════════════════════════ SDN model settings ════════════════════════ #

    sdn_model_args = deepcopy(common_args)
    sdn_model_args.with_backbone = False
    sdn_model_args.model_class = 'sdn'
    sdn_model_args.model_args = {}
    sdn_model_args.model_args.head_type = 'transformer_standard_head'
    sdn_model_args.model_args.place_at = list(range(NUM_ICS))
    sdn_model_args.optimizer_class = 'adam'
    sdn_model_args.optimizer_args = {}
    sdn_model_args.optimizer_args.lr = 0.01
    sdn_model_args.optimizer_args.weight_decay = 0.0001
    sdn_model_args.batch_size = EE_BATCH_SIZE
    sdn_model_args.epochs = 5
    sdn_model_args.eval_points = 20

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
    pbee_model_args.batch_size = EE_BATCH_SIZE
    pbee_model_args.epochs = 0
    pbee_model_args.eval_points = 20

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
    gpf_model_args.model_args.head_type = 'transformer_standard_head'
    gpf_model_args.model_args.place_at = list(range(NUM_ICS))
    gpf_model_args.model_args.head_dim = 768
    gpf_model_args.model_args.state_dropout = 0.0
    gpf_model_args.optimizer_args = {}
    gpf_model_args.optimizer_args.lr = 0.001
    gpf_model_args.optimizer_args.weight_decay = 0.0001
    gpf_model_args.batch_size = EE_BATCH_SIZE
    gpf_model_args.epochs = 5
    gpf_model_args.eval_points = 20

    # ════════════════════════ train GPF models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(gpf_model_args)
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
    display_names.append(f'GPF')
    # display_names.append(f'GPF lr={lr} wd={wd}')

    # ════════════════════════ L2W model settings ════════════════════════ #

    l2w_model_args = deepcopy(common_args)
    l2w_model_args.with_backbone = False
    l2w_model_args.model_class = 'l2w'
    l2w_model_args.model_args = {}
    l2w_model_args.model_args.head_type = 'transformer_standard_head'
    l2w_model_args.model_args.place_at = list(range(NUM_ICS))
    l2w_model_args.optimizer_class = 'sgd'
    l2w_model_args.optimizer_args = {}
    l2w_model_args.optimizer_args.lr = 0.5
    l2w_model_args.optimizer_args.momentum = 0.9
    l2w_model_args.optimizer_args.weight_decay = 0.0001
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
    l2w_model_args.batch_size = EE_BATCH_SIZE
    l2w_model_args.epochs = 5
    l2w_model_args.eval_points = 20

    # ════════════════════════ train L2W models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(l2w_model_args)
        args.exp_id = exp_id
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
    cascading_model_args.model_args.head_type = 'transformer_cascading_head'
    cascading_model_args.model_args.place_at = list(range(NUM_ICS))
    cascading_model_args.optimizer_class = 'adam'
    cascading_model_args.optimizer_args = {}
    cascading_model_args.optimizer_args.lr = 0.01
    cascading_model_args.optimizer_args.weight_decay = 0.0001
    cascading_model_args.batch_size = EE_BATCH_SIZE
    cascading_model_args.epochs = 5
    cascading_model_args.eval_points = 20

    # ════════════════════════ train ZTW cascading models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(cascading_model_args)
        args.exp_id = exp_id
        # args.optimizer_args.lr = lr
        # args.optimizer_args.weight_decay = wd
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
    # display_names.append(f'Cascading lr={lr} wd={wd}')
    cascading_exp_name = exp_name

    # ════════════════════════ ZTW ensembling model settings ════════════════════════ #

    ensembling_model_args = deepcopy(common_args)
    ensembling_model_args.with_backbone = False
    ensembling_model_args.model_class = 'ztw_ensembling'
    ensembling_model_args.model_args = {}
    ensembling_model_args.optimizer_class = 'adam'
    ensembling_model_args.optimizer_args = {}
    ensembling_model_args.optimizer_args.lr = 0.1
    ensembling_model_args.optimizer_args.weight_decay = 0.0
    ensembling_model_args.batch_size = EE_BATCH_SIZE
    ensembling_model_args.epochs = 5
    ensembling_model_args.eval_points = 20
    # ════════════════════════ train ZTW ensembling models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(ensembling_model_args)
        args.exp_id = exp_id
        args.base_on = cascading_exp_name
        base_run_name = f'{cascading_exp_name}_{exp_id}'
        dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
        executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
        job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    # display_names.append(f'ZTW o={optimizer} bs={bs} lr={lr} wd={wd}')
    display_names.append(f'ZTW')

    # ═════════════════════════════════════════════════════════ #

    print(f'Exp names: {exp_names}')
    print(f'Display names: {display_names}')
    print(f'SLURM JIDs: {[job.job_id for job in jobs]}')

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()
    out_dir_name = f'ztw_{MODEL.lower()}_{common_args.dataset}'
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
                               slurm_gpus_per_task=gpus_per_task,
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
                               slurm_gpus_per_task=gpus_per_task,
                               slurm_cpus_per_gpu=cpus_per_gpu,
                               slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={'dependency': dependency_str})
    submit_job(executor, cost_vs_table, plot_args)


if __name__ == '__main__':
    main()

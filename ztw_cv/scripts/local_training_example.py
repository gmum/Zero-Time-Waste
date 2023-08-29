import os
from pathlib import Path

from common import get_default_args
from train import train
from utils import generate_run_name


# command:
# python -m scripts.local_training_example


def main():
    args = get_default_args()

    args.exp_id = 1  # this also indicates the experiment id in wandb

    args.use_wandb = False
    args.runs_dir = Path(os.environ["RUNS_DIR"])
    args.dataset = "tinyimagenet"
    args.mixup_alpha = 0.0

    args.loss_type = "ce"
    args.loss_args = {}
    args.loss_args.label_smoothing = 0.0

    args.optimizer_class = "adam"
    args.optimizer_args = {}
    args.optimizer_args.lr = 0.01
    args.optimizer_args.weight_decay = 0.0005

    args.scheduler_class = "cosine"
    args.scheduler_args = {}

    args.batch_size = 128
    args.epochs = 100
    args.eval_points = 20

    # model
    args.model_class = "resnet50"
    args.model_args = {}
    args.model_args.num_classes = 200

    name = generate_run_name(args)[0]
    print("Run name:", name)

    # ════════════════════════ run the train function directly ════════════════════════ #

    train(args)

    # ════════════════════════ or use submitit local executor ════════════════════════ #

    # # use LocalExecutor to run job locally, without the need for SLURM cluster
    # executor = submitit.LocalExecutor(folder=os.environ["LOGS_DIR"])
    # executor.update_parameters(
    #     timeout_min=1,  # maximum time limit in minutes, after which the job will be cancelled
    #     cpus_per_task=4,
    #     cpus_per_node=4,
    #     gpus_per_task=0,
    #     gpus_per_node=0,
    #     # mem_per_gpu="4G",
    # )

    # job = executor.submit(train, args)
    # print("Current job:", job)

    # result = job.result()  # wait for job to finish
    # print("Result:", result)


if __name__ == "__main__":
    main()

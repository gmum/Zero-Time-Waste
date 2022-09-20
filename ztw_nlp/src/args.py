from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    ee_model: str
    base_model: str
    task: str
    seed: int

    max_epochs: int
    patience: Optional[int]
    lr: float
    train_batch_size: int
    val_batch_size: int

    ensembling: bool
    ensembling_lr: Optional[float]
    ensembling_epochs: int

    evaluate_ee_thresholds: bool
    early_exit_criterion: str

    model_path: Optional[str]
    output_dir: Path
    tags: Optional[str]
    num_workers: int
    device: str

    limit_train_batches: Optional[int]
    limit_val_batches: Optional[int]

    def __post_init__(self):
        self.task = self.task.lower()
        self.ee_model = self.ee_model.lower()
        self.base_model = self.base_model.lower()

        if self.ensembling and self.ensembling_lr is None:
            self.ensembling_lr = self.lr


def parse_args() -> Config:
    parser = ArgumentParser()
    setting = parser.add_argument_group("setting", "Base experiment settings.")
    setting.add_argument(
        "--ee_model",
        default="pabee",
        choices=["pabee", "sdn", "ztw"],
        help="Type of early exit model to train.",
    )
    setting.add_argument(
        "--base_model",
        default="bert_base",
        choices=["bert_base"],
        help="Base model to use.",
    )
    setting.add_argument("--task", type=str, required=True, help="Task name.")
    setting.add_argument("--seed", default=0, type=int, help="Random seed to use.")

    training = parser.add_argument_group("training", "Basic training hyperparameters.")
    training.add_argument("--max_epochs", default=50, type=int, help="Max train epochs.")
    training.add_argument(
        "--patience",
        default=None,
        type=int,
        help="If provided, will enable early stopping with this patience.",
    )
    training.add_argument("--lr", default=0.01, type=float, help="Model learning rate.")
    training.add_argument(
        "--train_batch_size",
        default=128,
        type=int,
        help="Batch size used for training.",
    )
    training.add_argument(
        "--val_batch_size",
        default=128,
        type=int,
        help="Batch size used for validation.",
    )

    ztw = parser.add_argument_group("ztw", "Hyperparameters related to ZTW training.")
    ztw.add_argument(
        "--ensembling",
        action="store_true",
        default=False,
        help="If set, will activate ensembling in ZTW. By default, only cascading is active.",
    )
    ztw.add_argument(
        "--ensembling_lr",
        type=float,
        default=None,
        help="Learning rate to set for ensembles (if autotuning lr is off).",
    )
    ztw.add_argument(
        "--ensembling_epochs",
        type=int,
        default=100,
        help="Number of epochs to train ensembles for.",
    )

    evaluation = parser.add_argument_group("evaluation", "Settings for evaluation")
    evaluation.add_argument(
        "--evaluate_ee_thresholds",
        action="store_true",
        default=False,
        help="Whether to run early exits evaluation",
    )
    evaluation.add_argument(
        "--early_exit_criterion",
        default="max_confidence",
        choices=["entropy", "patience", "max_confidence"],
        help="Criterion to use for early exit evaluation",
    )

    io = parser.add_argument_group("io", "In and out settings.")
    io.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to BERT model. Must be provided in `dataset_type` is `online`.",
    )
    io.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to directory with model saves.",
    )
    io.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Optional comma-separated tags assigned to the wandb run.",
    )
    io.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers to use for DataLoaders.",
    )
    io.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (`cuda` or `cpu`)",
    )

    debug = parser.add_argument_group("debug", "Settings for debugging.")
    debug.add_argument(
        "--limit_train_batches",
        default=None,
        type=int,
        help="If set, will limit number of train batches to this value.",
    )
    debug.add_argument(
        "--limit_val_batches",
        default=None,
        type=int,
        help="If set, will limit number of val batches to this value.",
    )

    args = parser.parse_args()
    return Config(**args.__dict__)

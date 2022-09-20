import os

import pytorch_lightning as pl
import torch.utils.data
from dotenv import load_dotenv
from pytorch_lightning.loggers.wandb import WandbLogger

from src.args import Config, parse_args
from src.consts import TASK_TO_VAL_SPLIT_NAME
from src.dataset import get_data_loaders, get_validation_data_loaders_for_ee
from src.models.model import EarlyExitModel
from src.utils import get_run_name, set_seed

load_dotenv()


def train(config: Config) -> None:
    set_seed(config.seed)

    train_data_loader, val_data_loader = get_data_loaders(config)

    tags = config.tags
    if tags is not None:
        tags = tags.split(",")
    logger = WandbLogger(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        name=get_run_name(config),
        tags=tags,
    )

    model = EarlyExitModel(config)
    checkpoint_path = _train(
        config=config,
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        max_epochs=config.max_epochs,
        logger=logger,
        monitor_metric=f"{TASK_TO_VAL_SPLIT_NAME[config.task]}/total_loss",
    )

    if config.ensembling:
        model.set_ensembles_on()
        checkpoint_path = _train(
            config=config,
            model=model,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            max_epochs=config.ensembling_epochs,
            logger=logger,
            monitor_metric=f"{TASK_TO_VAL_SPLIT_NAME[config.task]}_ensembling/total_loss",
        )

    if config.evaluate_ee_thresholds:
        model = EarlyExitModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
        val_ee_data_loader = get_validation_data_loaders_for_ee(config)
        model.log_final_metrics(val_ee_data_loader, TASK_TO_VAL_SPLIT_NAME[config.task])


def _train(
    config: Config,
    model: pl.LightningModule,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
    max_epochs: int,
    logger: pl.loggers.WandbLogger,
    monitor_metric: str,
) -> str:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.output_dir / get_run_name(config),
        monitor=monitor_metric,
    )
    callbacks = [
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelSummary(),
    ]
    if config.patience is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(monitor=monitor_metric, patience=config.patience)
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        gpus=1 if config.device == "cuda" else None,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    train(parse_args())

import io
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch.optim
import torchmetrics
from PIL import Image
from torch.nn.functional import cross_entropy, nll_loss
from tqdm import tqdm
from transformers.models.bert import BertForSequenceClassification

import wandb
from src.args import Config
from src.consts import TASK_TO_VAL_SPLIT_NAME
from src.models.ee.simple import SimpleEarlyExitModel
from src.models.ee.ztw import ZTWModel

sns.set_style("darkgrid")


class EarlyExitModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(asdict(config))

        self.config = config
        if config.base_model == "bert_base":
            num_hidden_layers = 12
            hidden_size = 768
            if config.task == "mnli":
                num_labels = 3
            else:
                num_labels = 2
        else:
            raise NotImplementedError()

        self.num_hidden_layers = num_hidden_layers
        self.ee_model = config.ee_model
        self.ensembles_active = False
        if config.ee_model == "pabee":
            self.model = SimpleEarlyExitModel(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                num_labels=num_labels,
            )
        elif config.ee_model == "sdn":
            self.model = SimpleEarlyExitModel(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                num_labels=num_labels,
            )
        elif config.ee_model == "ztw":
            self.model = ZTWModel(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                num_labels=num_labels,
            )
        else:
            raise NotImplementedError()

        self.bert = BertForSequenceClassification.from_pretrained(config.model_path, from_tf=False)
        self.bert.to(self.config.device)
        self.bert.eval()

        self.validation_key = TASK_TO_VAL_SPLIT_NAME[self.config.task]

    def configure_optimizers(self):
        if self.ensembles_active:
            lr = self.config.ensembling_lr
        else:
            lr = self.config.lr
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def set_ensembles_on(self):
        self.ensembles_active = True
        self.model.set_cascades(on=False)
        self.model.set_ensembles(on=True)
        self.validation_key = f"{TASK_TO_VAL_SPLIT_NAME[self.config.task]}_ensembling"

    def forward(self, x: torch.Tensor, ensemble: bool = False) -> torch.Tensor:
        return self.model(x, ensemble=ensemble)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._step(batch)

        self.log("train/loss", outputs["loss"])
        return outputs

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._epoch_end(outputs, "train")

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        split_name = self.validation_key
        self._epoch_end(outputs, split_name)

    def _step(
        self,
        batch: Any,
    ) -> Dict[str, torch.Tensor]:
        logits, states, labels = self._prepare_batch(batch)
        head_logits = self.forward(states, ensemble=self.ensembles_active)
        if self.ee_model == "ztw" and self.ensembles_active:
            head_losses = [nll_loss(logits, labels) for logits in head_logits]
        else:
            head_losses = [cross_entropy(logits, labels) for logits in head_logits]

        output = {
            "loss": sum(head_losses),
            **{f"head_{i + 1}_preds": preds.detach().cpu() for i, preds in enumerate(head_logits)},
            **{f"head_{i + 1}_loss": loss.detach().cpu() for i, loss in enumerate(head_losses)},
            "labels": labels.detach().cpu(),
        }
        return output

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            labels = batch.pop("label")
            model_outputs = self.bert(
                **{key: value.to(self.config.device) for key, value in batch.items()},
                output_hidden_states=True,
            )
            states = torch.stack(model_outputs.hidden_states[1:], dim=1)
            logits = model_outputs.logits

        return logits, states, labels

    def _epoch_end(self, outputs: List[Dict[str, torch.Tensor]], key: str) -> None:
        head_f1s = {i + 1: torchmetrics.F1Score() for i in range(self.num_hidden_layers)}
        head_accs = {i + 1: torchmetrics.Accuracy() for i in range(self.num_hidden_layers)}
        head_losses = {i + 1: torchmetrics.MeanMetric() for i in range(self.num_hidden_layers)}
        total_loss = torchmetrics.MeanMetric()
        for output in outputs:
            labels = output["labels"]
            for i in range(self.num_hidden_layers):
                preds = output[f"head_{i + 1}_preds"]
                loss = output[f"head_{i + 1}_loss"]

                head_f1s[i + 1].update(preds=preds, target=labels)
                head_accs[i + 1].update(preds=preds, target=labels)
                head_losses[i + 1].update(loss)
            total_loss.update(output["loss"].detach().cpu())

        head_f1s = {key: metric.compute() for key, metric in head_f1s.items()}
        head_accs = {key: metric.compute() for key, metric in head_accs.items()}
        head_losses = {key: metric.compute() for key, metric in head_losses.items()}
        for i in range(self.num_hidden_layers):
            self.log(f"{key}/head_f1/{i + 1}", head_f1s[i + 1])
            self.log(f"{key}/head_acc/{i + 1}", head_accs[i + 1])
            self.log(f"{key}/head_loss/{i + 1}", head_losses[i + 1])
        self.log(
            f"{key}/average_head_f1",
            sum(v for v in head_f1s.values()) / len(head_f1s),
        )
        self.log(
            f"{key}/average_head_acc",
            sum(v for v in head_accs.values()) / len(head_accs),
        )
        self.log(f"{key}/total_loss", total_loss.compute())

    def log_final_metrics(self, dataloader: torch.utils.data.DataLoader, key: str) -> None:
        if self.config.early_exit_criterion == "entropy":
            ee_thresholds = [
                round(t, 5)
                for t in np.concatenate(
                    [
                        np.linspace(0, 0.5, num=251),
                        np.linspace(0.51, 1.0, num=50),
                    ]
                )
            ]
        elif self.config.early_exit_criterion == "max_confidence":
            ee_thresholds = [
                round(t, 5)
                for t in np.concatenate(
                    [
                        np.linspace(0.5, 0.94, num=45),
                        np.linspace(0.95, 1.0, num=51),
                    ]
                )
            ]
        elif self.config.early_exit_criterion == "patience":
            ee_thresholds = [i for i in range(1, self.model.num_hidden_layers - 1)]
        else:
            raise NotImplementedError()

        self.eval()
        self.model.to(self.config.device)
        with torch.no_grad():
            avg_layer_per_thresh, accuracy_per_thresh = self._log_average_layer_vs_accuracy(
                key, ee_thresholds, dataloader
            )

            self._log_exit_layer_metrics(key, avg_layer_per_thresh, accuracy_per_thresh)
            self._log_accuracy_with_forced_exits_at_layers(key, dataloader)
            self._log_accuracy_integrated_over_avg_layer(
                key, avg_layer_per_thresh, accuracy_per_thresh
            )

    def _log_average_layer_vs_accuracy(
        self, key: str, ee_thresholds: List[float], dataloader: torch.utils.data.DataLoader
    ) -> Tuple[List[float], List[float]]:
        avg_layer_per_thresh = []
        accuracy_per_thresh = []
        for thresh in tqdm(
            ee_thresholds,
            desc="Running average layer vs accuracy evaluation...",
        ):
            avg_layer = torchmetrics.MeanMetric()
            accuracy = torchmetrics.Accuracy()
            for batch in dataloader:
                _, states, labels = self._prepare_batch(batch)
                logits = self.model.forward_early_exit(
                    states,
                    ee_criterion=self.config.early_exit_criterion,
                    ee_threshold=thresh,
                )
                avg_layer.update(len(logits))
                preds = logits[-1].argmax(dim=-1)
                if len(preds.shape) == 0:
                    preds = preds.unsqueeze(0)
                if len(labels.shape) == 0:
                    labels = labels.unsqueeze(0)
                accuracy.update(preds=preds.cpu(), target=labels.cpu())
            avg_layer_per_thresh.append(avg_layer.compute().item() / self.model.num_hidden_layers)
            accuracy_per_thresh.append(accuracy.compute().item())

        layer_vs_acc_data = [
            [layer, acc] for (layer, acc) in zip(avg_layer_per_thresh, accuracy_per_thresh)
        ]
        layer_vs_acc_table = wandb.Table(
            data=layer_vs_acc_data, columns=["Average_layer", "Accuracy"]
        )
        wandb.log(
            {
                f"{key}/avg_layer_vs_accuracy": wandb.plot.scatter(
                    layer_vs_acc_table,
                    "Average_layer",
                    "Accuracy",
                    title="Average layer vs accuracy.",
                )
            }
        )

        return avg_layer_per_thresh, accuracy_per_thresh

    @staticmethod
    def _log_exit_layer_metrics(
        key: str, avg_layer_per_thresh: List[float], accuracy_per_thresh: List[float]
    ) -> None:
        layer_thresholds = (0.25, 0.5, 0.75, 1.0)
        actual_thresholds = []
        layer_accuracies = []
        for layer_thresh in (0.25, 0.5, 0.75, 1.0):
            thresh_layer, thresh_acc = get_avg_layer_acc_for_thresh(
                avg_layer_per_thresh, accuracy_per_thresh, layer_thresh
            )
            actual_thresholds.append(thresh_layer)
            layer_accuracies.append(thresh_acc)
        exit_data = [
            [thresh, exited_layer]
            for (thresh, exited_layer) in zip(layer_thresholds, actual_thresholds)
        ]
        exit_table = wandb.Table(data=exit_data, columns=["Thresh", "Actual exit"])
        wandb.log(
            {
                f"{key}/exit_threshold_vs_exit_layer": wandb.plot.scatter(
                    exit_table,
                    "Thresh",
                    "Actual exit",
                    title="Exit threshold vs actual exit layer.",
                )
            }
        )
        acc_data = [
            [thresh, exit_acc] for (thresh, exit_acc) in zip(layer_thresholds, layer_accuracies)
        ]
        acc_table = wandb.Table(data=acc_data, columns=["Thresh", "Accuracy"])
        wandb.log(
            {
                f"{key}/exit_threshold_vs_accuracy": wandb.plot.scatter(
                    acc_table,
                    "Thresh",
                    "Accuracy",
                    title="Exit threshold vs accuracy.",
                )
            }
        )

    def _log_accuracy_with_forced_exits_at_layers(
        self, key: str, dataloader: torch.utils.data.DataLoader
    ) -> None:
        layer_indices = []
        per_layer_accuracies = []
        for layer_idx in tqdm(
            list(range(self.model.num_hidden_layers)),
            desc="Running accuracy evaluation on concrete layers...",
        ):
            layer_accuracy = torchmetrics.Accuracy()
            for batch in dataloader:
                _, states, labels = self._prepare_batch(batch)
                logits = self.model.forward_force_exit(
                    states,
                    force_exit=layer_idx,
                    ensemble=self.config.ensembling,
                )
                preds = logits[-1].argmax(dim=-1)
                if len(preds.shape) == 0:
                    preds = preds.unsqueeze(0)
                if len(labels.shape) == 0:
                    labels = labels.unsqueeze(0)
                layer_accuracy.update(preds=preds.cpu(), target=labels.cpu())
            layer_indices.append(layer_idx + 1)
            per_layer_accuracies.append(layer_accuracy.compute().item())

        average_layer_vs_accuracy_data = [
            [layer_idx, layer_acc]
            for (layer_idx, layer_acc) in zip(layer_indices, per_layer_accuracies)
        ]
        average_layer_vs_accuracy_table = wandb.Table(
            data=average_layer_vs_accuracy_data, columns=["Layer idx", "Accuracy"]
        )
        wandb.log(
            {
                f"{key}/exit_layer_vs_accuracy": wandb.plot.scatter(
                    average_layer_vs_accuracy_table,
                    "Exit layer",
                    "Accuracy",
                    title="Exit layer vs accuracy.",
                )
            }
        )

    def _log_accuracy_integrated_over_avg_layer(
        self, key: str, avg_layer_per_thresh: List[float], accuracy_per_thresh: List[float]
    ) -> None:
        layer_vs_accuracy = list(zip(avg_layer_per_thresh, accuracy_per_thresh))
        layer_vs_accuracy = sorted(layer_vs_accuracy, key=lambda pair: pair[0])

        integrated_accuracy = 0.0
        for p1, p2 in zip(layer_vs_accuracy, layer_vs_accuracy[1:]):
            avg_layer_step = p2[0] - p1[0]
            avg_accuracy_height = (p1[1] + p2[1]) / 2
            integrated_accuracy += avg_accuracy_height * avg_layer_step
        wandb.log({f"{key}/integrated_accuracy": integrated_accuracy})

    def _log_ensemble_params(self) -> None:
        if self.ensembles_active:
            ensemble_weights = torch.zeros(
                (self.model.num_hidden_layers, self.model.num_hidden_layers)
            )
            ensemble_biases = torch.zeros((self.num_hidden_layers, self.model.num_labels))
            for i, ensemble in enumerate(self.model.ensembles):
                ensemble_weights[i, : i + 1] = ensemble._weight.data.detach().cpu()
                ensemble_biases[i] = ensemble._bias.detach().cpu()

            plt.cla()
            plot = _create_heatmap(
                ensemble_weights, "Ensemble weight values", "Previous head logits idx", "Head idx"
            )
            _log_figure(plot, "ensemble_params/weights")

            plt.cla()
            plot = _create_heatmap(ensemble_biases, "Ensemble bias values", "Class idx", "Head idx")
            _log_figure(plot, "ensemble_params/biases")


def get_avg_layer_acc_for_thresh(
    avg_layers: List[float], accuracies: List[float], thresh: float
) -> Tuple[float, float]:
    layer_acc = [
        (avg_layer, acc) for avg_layer, acc in zip(avg_layers, accuracies) if avg_layer < thresh
    ]
    if len(layer_acc) > 0:
        return max(layer_acc, key=lambda pair: pair[0])
    else:
        return thresh, 0.0


def _create_heatmap(data: torch.Tensor, title: str, x_label: str, y_label: str) -> plt.Axes:
    plt.cla()
    plt.clf()
    plot = sns.heatmap(np.array(data), annot=True, fmt=".1f", cmap="summer")
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    return plot


def _log_figure(plot: plt.Axes, key: str) -> None:
    fig = plot.get_figure()
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches="tight")
    with Image.open(buffer) as img:
        img = wandb.Image(img)
    wandb.log({key: img})

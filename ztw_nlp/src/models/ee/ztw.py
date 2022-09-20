from typing import List, Union

import torch

from src.models.ee.base import BaseEarlyExitModel, would_early_exit


class ZTWModel(BaseEarlyExitModel):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.0,
    ):
        """
        ZTW model.
        """
        super().__init__(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.classifiers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_size, num_labels)
                if i == 0
                else torch.nn.Linear(hidden_size + num_labels, num_labels)
                for i in range(num_hidden_layers)
            ]
        )
        self.detach_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(num_labels) for _ in range(num_hidden_layers)]
        )
        self.ensembles = torch.nn.ModuleList(
            [WeightedEnsemble(i + 1, self.num_labels) for i in range(self.num_hidden_layers)]
        )

        self.exit_layer = None

    def forward(self, hidden_states: torch.Tensor, ensemble: bool = True) -> List[torch.Tensor]:
        layer_logits = []
        prev_logits: List[torch.Tensor] = []
        for layer_idx in range(self.num_hidden_layers):
            logits = self.forward_single_layer(
                hidden_states[:, layer_idx, :, :],
                prev_logits=prev_logits,
                layer_idx=layer_idx,
                ensemble=ensemble,
            )
            layer_logits.append(logits)
            prev_logits.append(self.detach_norms[layer_idx](logits.detach()))

            if self.exit_layer is not None and layer_idx == self.exit_layer:
                break
        return layer_logits

    def forward_single_layer(
        self,
        x: torch.Tensor,
        layer_idx: int,
        prev_logits: List[torch.Tensor],
        ensemble: bool = False,
    ) -> torch.Tensor:
        x = self.dropout(x)
        x = self.poolers[layer_idx](x).squeeze(axis=1)
        if len(prev_logits) > 0:
            x = torch.cat([x, prev_logits[-1]], dim=-1)
        x = self.classifiers[layer_idx](x)
        if ensemble:
            x = torch.stack(prev_logits + [x], dim=1)
            x = self.ensembles[layer_idx](x)
        return x

    def forward_force_exit(
        self,
        hidden_states: torch.Tensor,
        force_exit: int,
        ensemble: bool = False,
    ) -> List[torch.Tensor]:
        if force_exit < 0 or force_exit > self.num_hidden_layers - 1:
            raise ValueError(f"`force_exit` should be between 0 and {self.num_hidden_layers - 1}")

        layer_logits = []
        prev_logits: List[torch.Tensor] = []
        for layer_idx in range(self.num_hidden_layers):
            logits = self.forward_single_layer(
                hidden_states[:, layer_idx, :, :],
                prev_logits=prev_logits,
                layer_idx=layer_idx,
                ensemble=ensemble,
            )
            layer_logits.append(logits)
            prev_logits.append(self.detach_norms[layer_idx](logits.detach()))

            if force_exit == layer_idx:
                return layer_logits

        raise ValueError("Didn't manage to early exit, this most likely indicates a bug.")

    def forward_early_exit(
        self,
        hidden_states: torch.Tensor,
        ee_criterion: str,
        ee_threshold: Union[int, float],
        ensemble: bool = False,
    ) -> List[torch.Tensor]:
        layer_logits = []
        prev_logits: List[torch.Tensor] = []
        for layer_idx in range(self.num_hidden_layers):
            logits = self.forward_single_layer(
                hidden_states[:, layer_idx, :, :],
                prev_logits=prev_logits,
                layer_idx=layer_idx,
                ensemble=ensemble,
            )
            layer_logits.append(logits)
            prev_logits.append(self.detach_norms[layer_idx](logits.detach()))

            if would_early_exit(layer_logits, ee_criterion, ee_threshold):
                return layer_logits

        return layer_logits

    def set_cascades(self, on: bool):
        for param in self.ensembles.parameters():
            param.requires_grad = on
        for param in self.detach_norms.parameters():
            param.requires_grad = on

    def set_ensembles(self, on: bool):
        for param in self.poolers.parameters():
            param.requires_grad = on
        for param in self.classifiers.parameters():
            param.requires_grad = on


class WeightedEnsemble(torch.nn.Module):
    def __init__(self, num_heads: int, num_classes: int):
        super().__init__()
        self._num_heads = num_heads
        self._num_classes = num_classes

        w = torch.normal(0, 0.01, size=(1, num_heads, 1))
        self._weight = torch.nn.Parameter(w)
        self._bias = torch.nn.Parameter(torch.zeros(size=(num_classes,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x are activations from the heads
        # x shape is (batch_size, num_heads, num_classes)
        x = torch.log_softmax(x, dim=-1)
        exp_weight = torch.exp(self._weight)
        x = torch.mean(x * exp_weight, dim=1) + self._bias
        return x

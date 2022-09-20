from typing import List, Union

import torch

from src.models.ee.base import BaseEarlyExitModel, would_early_exit


class SimpleEarlyExitModel(BaseEarlyExitModel):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.0,
    ):
        """
        Simple early exit model obtained by attaching poolers and linear classifiers to each
        intermediate layer of the model. Can be used either as PABEE or as SDN
        """
        super().__init__(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, num_labels) for _ in range(num_hidden_layers)]
        )

        self.exit_layer = None

    def forward(self, hidden_states: torch.Tensor, ensemble: bool = False) -> List[torch.Tensor]:
        layer_logits = []
        for layer_idx in range(self.num_hidden_layers):
            logits = self.forward_single_layer(
                hidden_states[:, layer_idx, :, :], layer_idx=layer_idx
            )
            layer_logits.append(logits)

            if self.exit_layer is not None and layer_idx == self.exit_layer:
                break
        return layer_logits

    def forward_single_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = self.dropout(x)
        x = self.poolers[layer_idx](x).squeeze()
        return self.classifiers[layer_idx](x)

    def forward_force_exit(
        self,
        hidden_states: torch.Tensor,
        force_exit: int,
        ensemble: bool = False,
    ) -> List[torch.Tensor]:
        if force_exit < 0 or force_exit > self.num_hidden_layers - 1:
            raise ValueError(f"`force_exit` should be between 0 and {self.num_hidden_layers - 1}")

        layer_logits = []
        for layer_idx in range(self.num_hidden_layers):
            logits = self.forward_single_layer(
                hidden_states[:, layer_idx, :, :], layer_idx=layer_idx
            )
            layer_logits.append(logits)

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
        for layer_idx in range(self.num_hidden_layers):
            logits = self.forward_single_layer(
                hidden_states[:, layer_idx, :, :], layer_idx=layer_idx
            )
            layer_logits.append(logits)

            if would_early_exit(layer_logits, ee_criterion, ee_threshold):
                return layer_logits

        return layer_logits

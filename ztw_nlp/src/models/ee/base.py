from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.nn

from src.models.bert_pooler import BertPooler


class BaseEarlyExitModel(ABC, torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.0,
    ):
        """
        Base early exit model with dropout, poolers and forward methods for early exit scenarios.
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.dropout = torch.nn.Dropout(dropout)
        self.poolers = torch.nn.ModuleList(
            [BertPooler(hidden_size) for _ in range(num_hidden_layers)]
        )

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, ensemble: bool = False) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def forward_force_exit(
        self,
        hidden_states: torch.Tensor,
        force_exit: int,
        ensemble: bool = False,
    ) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def forward_early_exit(
        self,
        hidden_states: torch.Tensor,
        ee_criterion: str,
        ee_threshold: Union[int, float],
        ensemble: bool = False,
    ) -> List[torch.Tensor]:
        ...


def would_early_exit(
    layer_logits: List[torch.Tensor],
    ee_criterion: str,
    ee_threshold: Union[int, float],
) -> bool:
    if ee_criterion == "entropy":
        prob = torch.nn.functional.softmax(layer_logits[-1], dim=-1)
        entropy = -torch.sum(prob * torch.log(prob), dim=-1)
        if torch.all(entropy < ee_threshold):
            return True
    elif ee_criterion == "max_confidence":
        prob = torch.nn.functional.softmax(layer_logits[-1], dim=-1)
        max_confidence = torch.max(prob, dim=-1).values
        if torch.all(max_confidence > ee_threshold):
            return True
    elif ee_criterion == "patience":
        current_pred = layer_logits[-1].argmax(dim=-1)
        prev_logits = layer_logits[-int(1 + ee_threshold) : -1]
        if len(prev_logits) < ee_threshold:
            return False

        prev_preds = [logits.argmax(dim=-1) for logits in prev_logits]
        if all(pred == current_pred for pred in prev_preds):
            return True
    else:
        raise ValueError(f"Unknown early exit criterion: {ee_criterion}")

    return False

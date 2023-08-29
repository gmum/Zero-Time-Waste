import logging
from collections import defaultdict
from math import sqrt
from typing import Type, Set

import torch
from torch import nn

from architectures.early_exits.base import EarlyExitingBase
from architectures.early_exits.heads import HEAD_TYPES, TransformerHead, CNNHead


class GPF(EarlyExitingBase):

    def __init__(self, base_model: nn.Module, head_type: Type, place_at: Set[int], head_dim,
                 state_dropout: float = 0.0):
        super().__init__()
        self._base_model = base_model
        self._place_at = place_at
        head_class = HEAD_TYPES[head_type]
        self._head_dim = head_dim
        self._heads = nn.ModuleList()
        self._imitators = nn.ModuleList()
        base_model_device = next(base_model.parameters()).device
        if hasattr(self._base_model, "input_sequences"):
            x_input = {
                k: torch.ones((1, v), device=base_model_device, dtype=torch.int) for k, v in self._base_model.input_sequences.items()
            }
        else:
            input_size = self._base_model.input_size if hasattr(self._base_model, 'input_size') else 32
            channels = self._base_model.input_channels if hasattr(self._base_model, 'input_channels') else 3
            x_input = torch.randn(1, channels, input_size, input_size, device=base_model_device)
        fg = self._base_model.forward_generator(x_input)
        i, x = 0, None
        in_size_list = []
        while True:
            try:
                x, _ = fg.send(x)
            except StopIteration:
                break
            if i in self._place_at:
                if issubclass(head_class, TransformerHead):
                    in_size_list.append(x.size(2))
                elif issubclass(head_class, CNNHead):
                    in_size_list.append(x.size(1))
            i += 1
        for in_size in in_size_list:
            self._heads.append(torch.nn.Sequential(head_class(in_size=in_size, out_size=head_dim),
                                                   nn.ReLU()))
            self._imitators.append(torch.nn.Sequential(torch.nn.Linear(head_dim, head_dim), nn.ReLU()))
        assert self.number_of_attached_heads == len(place_at), f'{place_at=} {i=}'
        self._reduction_layer_weight = torch.nn.Parameter(torch.zeros(self.number_of_attached_heads,
                                                                      head_dim,
                                                                      self.number_of_classes)
                                                          .normal_(mean=0.0, std=1 / sqrt(head_dim)))
        self._reduction_layer_bias = torch.nn.Parameter(torch.zeros(self.number_of_attached_heads,
                                                                    self.number_of_classes))
        self._gate_layer_weight = torch.nn.Parameter(torch.zeros(self.number_of_attached_heads,
                                                                 head_dim,
                                                                 self.number_of_classes)
                                                     .normal_(mean=0.0, std=1 / sqrt(head_dim)))
        self._gate_layer_bias = torch.nn.Parameter(torch.zeros(self.number_of_attached_heads,
                                                               self.number_of_classes))
        self._adaptive_balance = torch.nn.Sequential(
            nn.ReLU(),
            torch.nn.Linear(self.number_of_classes, self.number_of_classes),
            nn.Sigmoid())
        self._state_dropout = state_dropout
        self._cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')

    @property
    def number_of_attached_heads(self):
        return len(self._heads)

    @property
    def number_of_classes(self):
        return self._base_model.number_of_classes

    @property
    def _head_modules(self):
        raise RuntimeError('A single head module is not defined for GPF.')

    @property
    def _core_modules(self):
        return [self._base_model]

    def get_merged_layer_state(self, states, head_idx):
        # inconsistent with the description in the paper, but consistent with the original implementation
        # see https://github.com/lancopku/Early-Exit/blob/main/model/modeling_bert.py#L477-L503
        states = torch.stack(states, dim=1)
        # states is of (batch_size, num_heads, hidden_dim) size
        states = nn.functional.dropout(states, self._state_dropout).unsqueeze(-2)
        states_proj = (states @ self._reduction_layer_weight).squeeze(-2) + self._reduction_layer_bias
        past_current_state = torch.sum(states_proj[:, :head_idx + 1], dim=1)
        if head_idx < self.number_of_attached_heads:
            future_state = torch.sum(states_proj[:, head_idx + 1:], dim=1)
            gate_proj = (states[:, :head_idx + 1] @ self._gate_layer_weight[:head_idx + 1]).squeeze(-2) \
                        + self._gate_layer_bias[:head_idx + 1]
            prev_gate = torch.mean(gate_proj, dim=1)
            balance = self._adaptive_balance(prev_gate)
            merged_state = balance * past_current_state + (1 - balance) * future_state
        else:
            merged_state = past_current_state
        return merged_state

    def forward_generator(self, x_input: torch.Tensor):
        # TODO do not store activations as model state
        self.past_states = []
        self.future_states = defaultdict(list)
        fg = self._base_model.forward_generator(x_input)
        head_idx, i, x = 0, 0, None
        while True:
            try:
                x, final_output = fg.send(x)
            except StopIteration:
                break
            if final_output is not None and x is None:
                yield None, final_output
            elif i in self._place_at:
                state_i = self._heads[head_idx](x)
                for future_head_idx, imitator in enumerate(self._imitators):
                    if future_head_idx > head_idx:
                        approx_state_j = imitator(state_i)
                        self.future_states[head_idx].append(approx_state_j)
                head_output = self.get_merged_layer_state(self.past_states + [state_i] + self.future_states[head_idx],
                                                          head_idx)
                self.past_states.append(state_i.detach())
                head_idx += 1
                x = yield x, head_output
            i += 1

    def aux_loss(self):
        i_loss = 0.0
        device = next(self._heads.parameters()).device
        # the last head has none predicted future states
        for head_idx in range(self.number_of_attached_heads - 1):
            imitation_loss = 0.0
            actual_states = self.past_states[head_idx + 1:]
            predicted_future_states = self.future_states[head_idx]
            actual_states = torch.cat(actual_states, dim=0)
            predicted_future_states = torch.cat(predicted_future_states, dim=0)
            imitation_loss += self._cosine_loss(actual_states, predicted_future_states,
                                                torch.ones(predicted_future_states.size(0))
                                                .to(device))
            i_loss += imitation_loss
        i_loss /= (self.number_of_attached_heads - 1)
        return i_loss

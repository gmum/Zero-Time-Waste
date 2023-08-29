from abc import ABC, abstractmethod
from typing import Iterator, Union

import torch
from torch import nn
from torch.nn import functional as F


class EarlyExitingBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._mode = 'all'
        self._confidence_threshold = None
        self._selected_head = -1

    @property
    def number_of_heads(self):
        # one more for the original output
        return self.number_of_attached_heads + 1

    @property
    @abstractmethod
    def number_of_attached_heads(self):
        ...

    @property
    @abstractmethod
    def number_of_classes(self):
        ...

    @property
    @abstractmethod
    def _head_modules(self):
        ...

    @property
    @abstractmethod
    def _core_modules(self):
        ...

    @abstractmethod
    def forward_generator(self, x_input: torch.Tensor):
        ...

    def train(self, mode: Union[str, bool] = 'all'):
        if isinstance(mode, bool):
            nn.Module.train(self, mode)
            return
        else:
            self.training = mode
        if mode == 'all':
            nn.Module.train(self)
            self.requires_grad_(True)
        elif mode == 'without_backbone':
            nn.Module.train(self)
            self.requires_grad_(True)
            for m in self._core_modules:
                m.eval()
                m.requires_grad_(False)
        elif mode == 'only_selected_head':
            nn.Module.eval(self)
            self.requires_grad_(True)
            if self._selected_head >= 0 and self._selected_head < self.number_of_attached_heads:
                for p in self._head_modules[self._selected_head].parameters():
                    p.requires_grad = True
                self._head_modules[self._selected_head].train()
        elif mode == 'none':
            self.eval()
            self.requires_grad_(False)
        else:
            raise ValueError(f'Invalid mode value: {mode}')

    def head_modules(self):
        return self._head_modules

    def unfrozen_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad == True)

    @property
    def mode(self):
        return self._mode

    def set_threshold(self, threshold: float):
        self._mode = 'threshold'
        self._confidence_threshold = threshold
        self._selected_head = -1
        self._patience = -1

    def set_patience(self, patience: int):
        self._mode = 'patience'
        self._patience = patience
        self._confidence_threshold = 1.0
        self._selected_head = -1

    def select_head(self, head_idx: int):
        assert 0 <= head_idx < self.number_of_heads
        self._mode = 'selected'
        self._selected_head = head_idx
        self._confidence_threshold = 1.0
        self._patience = -1

    def all_mode(self):
        self._mode = 'all'

    def forward(self, x_input: torch.Tensor):
        assert self._mode in {'selected', 'threshold', 'all', 'patience'}
        enable_grad = True if self.training else False
        with torch.set_grad_enabled(enable_grad):
            fg, x, head_outputs = self.forward_generator(x_input), None, []
            if self._mode == 'threshold':
                sample_exited_at = torch.zeros(x_input.size(0), dtype=torch.int) - 1
                # TODO possibly use a tensor instead of a list
                sample_outputs = [torch.Tensor() for _ in range(x_input.size(0))]
                try:
                    head_idx = 0
                    while True:
                        x, head_output = fg.send(x)
                        head_outputs.append(head_output)
                        head_probs = F.softmax(head_output, dim=-1)
                        head_confidences, _ = head_probs.max(dim=-1)
                        # early exiting masks
                        unresolved_samples_mask = sample_exited_at == -1
                        exit_mask_local = (head_confidences > self._confidence_threshold).cpu().squeeze(dim=-1)
                        exit_mask_global = torch.zeros_like(unresolved_samples_mask, dtype=torch.bool)
                        exit_mask_global[unresolved_samples_mask] = exit_mask_local
                        # update sample head index array
                        sample_exited_at[exit_mask_global] = head_idx
                        # update sample return list
                        exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
                        exit_indices_local = exit_mask_local.nonzero().view(-1).tolist()
                        assert len(exit_indices_global) == len(exit_indices_local), \
                            f'exit_indices_global: {exit_indices_global} exit_indices_local: {exit_indices_local}'
                        for j, k in zip(exit_indices_global, exit_indices_local):
                            sample_outputs[j] = head_output[k]
                        # head handled
                        head_idx += 1
                        # continue only if there are unresolved samples
                        if (exit_mask_local).all():
                            break
                        # continue only with remaining sample subset
                        x = x[~exit_mask_local]
                except StopIteration:
                    exit_mask_global = unresolved_samples_mask
                    # update sample head index array
                    sample_exited_at[exit_mask_global] = head_idx
                    # update sample return list
                    exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
                    for j, k in enumerate(exit_indices_global):
                        sample_outputs[k] = head_output[j]
                outputs = torch.stack(sample_outputs)
                return outputs, head_outputs
            elif self._mode == 'patience':
                # TODO
                raise NotImplementedError()
            elif self._mode == 'selected':
                head_idx = 0
                while True:
                    try:
                        x, head_output = fg.send(x)
                        head_outputs.append(head_output)
                    except StopIteration:
                        raise RuntimeError('Selected head incorrectly set')
                    if self._selected_head == head_idx:
                        return head_output, head_outputs
                    head_idx += 1
            elif self._mode == 'all':
                while True:
                    try:
                        x, head_output = fg.send(x)
                    except StopIteration:
                        assert len(head_outputs) == self.number_of_heads
                        return head_outputs
                    head_outputs.append(head_output)
            else:
                raise ValueError('Wrong self._mode value!')

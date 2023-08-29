from typing import Type, Set

import torch
from torch import nn

from architectures.early_exits.base import EarlyExitingBase
from architectures.early_exits.heads import HEAD_TYPES, ViTHead, CNNHead, SwinHead


class ZTWCascading(EarlyExitingBase):
    def __init__(self, base_model: nn.Module, head_type: Type, place_at: Set[int], layer_norm: bool = True,
                 detach: bool = True):
        super().__init__()
        self._base_model = base_model
        self._place_at = place_at
        head_class = HEAD_TYPES[head_type]
        self._heads = nn.ModuleList()
        input_size = self._base_model.input_size if hasattr(self._base_model, 'input_size') else 32
        channels = self._base_model.input_channels if hasattr(self._base_model, 'input_channels') else 3
        base_model_device = next(base_model.parameters()).device
        x_input = torch.randn(1, channels, input_size, input_size, device=base_model_device)
        self._base_model.eval()
        # without the above eval() method call batch normalization statistics are updated in the forward below
        fg = self._base_model.forward_generator(x_input)
        i, x, cascading = 0, None, False
        while True:
            try:
                x, _ = fg.send(x)
            except StopIteration:
                break
            if i in self._place_at:
                assert x is not None, f'{place_at=} {i=}'
                if issubclass(head_class, CNNHead):
                    in_size = x.size(1)
                elif issubclass(head_class, ViTHead):
                    in_size = x.size(2)
                elif issubclass(head_class, SwinHead):
                    in_size = x.size(-1)
                self._heads.append(head_class(in_size=in_size, out_size=self.number_of_classes,
                                              cascading=cascading, layer_norm=layer_norm, detach=detach))
                cascading = True
            i += 1
        assert self.number_of_attached_heads == len(place_at), f'{place_at=} {i=}'

    @property
    def number_of_attached_heads(self):
        return len(self._heads)

    @property
    def number_of_classes(self):
        return self._base_model.number_of_classes

    @property
    def _head_modules(self):
        return self._heads

    @property
    def _core_modules(self):
        return [self._base_model]

    def forward_generator(self, x_input: torch.Tensor):
        fg = self._base_model.forward_generator(x_input)
        head_idx, i, x, prev_head_output = 0, 0, None, None
        while True:
            try:
                x, final_output = fg.send(x)
            except StopIteration:
                break
            if final_output is not None and x is None:
                yield None, final_output
            elif i in self._place_at:
                head_output = self._heads[head_idx](x, prev_head_output)
                prev_head_output = head_output
                x = yield x, head_output
                head_idx += 1
            i += 1


class GeometricWeightedEnsemble(nn.Module):
    def __init__(self, num_heads: int, num_classes: int):
        super().__init__()
        self._num_heads = num_heads
        self._num_classes = num_classes
        self._weight = nn.Parameter(torch.normal(0, 0.01, size=(1, num_heads, 1)))
        self._bias = nn.Parameter(torch.zeros(size=(1, num_classes,)))

    def forward(self, x: torch.Tensor):
        # x are logprobs from the heads
        # x shape is (batch_size, num_heads, num_classes)
        x = torch.mean(x * self._weight.exp(), dim=1) + self._bias
        return x


class ArithmeticWeightedEnsemble(nn.Module):
    EPS = 1e-40

    def __init__(self, num_heads: int, num_classes: int):
        super().__init__()
        self._num_heads = num_heads
        self._num_classes = num_classes
        self._weight = nn.Parameter(torch.normal(0, 0.01, size=(1, num_heads, 1)))
        self._bias = nn.Parameter(torch.zeros(size=(1, num_classes,)))

    def forward(self, x: torch.Tensor):
        # x are logprobs from the heads
        # x shape is (batch_size, num_heads, num_classes))
        x = x.exp()
        x = (x * self._weight.exp()).sum(dim=1) + self._bias.exp()
        x = x / x.sum(dim=1, keepdim=True)
        # back into logspace
        x = (x + self.EPS).log()
        return x


class ZTWEnsembling(EarlyExitingBase):
    def __init__(self, base_model: EarlyExitingBase, type: str = 'geometric'):
        super().__init__()
        self._base_model = base_model
        self._ensembles = nn.ModuleList()
        self._type = type
        # ZTW does not build an ensemble on the original (backbone model's) head
        for i in range(self._base_model.number_of_attached_heads):
            if self._type == 'geometric':
                self._ensembles.append(
                    GeometricWeightedEnsemble(num_heads=i + 1, num_classes=self._base_model.number_of_classes))
            elif self._type == 'arithmetic':
                self._ensembles.append(
                    ArithmeticWeightedEnsemble(num_heads=i + 1, num_classes=self._base_model.number_of_classes))

    @property
    def number_of_attached_heads(self):
        return len(self._ensembles)

    @property
    def number_of_classes(self):
        return self._base_model.number_of_classes

    @property
    def _head_modules(self):
        return self._ensembles

    @property
    def _core_modules(self):
        return [self._base_model]

    def forward_generator(self, x_input: torch.Tensor):
        fg = self._base_model.forward_generator(x_input)
        x, head_logprobs = None, []
        while True:
            try:
                x, head_output = fg.send(x)
            except StopIteration:
                break
            if x is None:
                yield None, head_output
            else:
                head_logprobs.append(torch.log_softmax(head_output, dim=-1))
                ensemble_output = self._ensembles[len(head_logprobs) - 1](torch.stack(head_logprobs, dim=1))
                x = yield x, ensemble_output

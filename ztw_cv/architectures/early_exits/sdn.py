from typing import Set

import torch
from torch import nn

from architectures.early_exits.base import EarlyExitingBase
from architectures.early_exits.heads import HEAD_TYPES, ViTHead, CNNHead, SwinHead


class SDN(EarlyExitingBase):
    def __init__(self, base_model: nn.Module, head_type: str, place_at: Set[int]):
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
        i, x = 0, None
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
                self._heads.append(head_class(in_size=in_size, out_size=self.number_of_classes))
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
        head_idx, i, x = 0, 0, None
        while True:
            try:
                x, final_output = fg.send(x)
            except StopIteration:
                break
            if final_output is not None and x is None:
                yield None, final_output
            elif i in self._place_at:
                head_output = self._heads[head_idx](x)
                x = yield x, head_output
                head_idx += 1
            i += 1

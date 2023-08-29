import torch

from architectures.early_exits.base import EarlyExitingBase


class PBEE(EarlyExitingBase):
    def __init__(self, base_model: EarlyExitingBase):
        super().__init__()
        self._base_model = base_model

    @property
    def number_of_attached_heads(self):
        return self._base_model.number_of_attached_heads

    @property
    def number_of_classes(self):
        return self._base_model.number_of_classes

    @property
    def _head_modules(self):
        return self._base_model._head_modules

    @property
    def _core_modules(self):
        return self._base_model._core_modules

    def forward_generator(self, x_input: torch.Tensor):
        yield from self._base_model.forward_generator(x_input)
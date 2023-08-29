from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SDNPool(nn.Module):
    def __init__(self, target_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self._alpha = nn.Parameter(torch.rand(1))
        self._max_pool = nn.AdaptiveMaxPool2d(target_size)
        self._avg_pool = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        # SDN code did not use sigmoid here, while it was used as proposed in:
        # https://arxiv.org/pdf/1509.08985.pdf
        # TODO check results with sigmoid
        avg_p = self._alpha * self._max_pool(x)
        max_p = (1 - self._alpha) * self._avg_pool(x)
        mixed = avg_p + max_p
        return mixed


class CNNHead(nn.Module):
    """
        Base class for all CNN heads.
    """

    def __init__(self):
        super().__init__()


class CNNStandardHead(CNNHead):
    def __init__(self, in_size: int, out_size: int, pool_size: int = 4):
        super().__init__()
        self._out_size = out_size
        self._pooling = SDNPool(pool_size)
        self._fc = nn.Linear(in_size * pool_size ** 2, out_size)

    def forward(self, x: torch.Tensor):
        x = F.relu(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x


class ConvHead(CNNHead):
    def __init__(self, in_size: int, out_size: int, pool_size: int = 4):
        super().__init__()
        self._out_size = out_size
        channels = in_size // 2
        self._conv = nn.Conv2d(in_size, channels, kernel_size=3, stride=2)
        self._pooling = SDNPool(pool_size)
        self._fc = nn.Linear(channels * pool_size ** 2, out_size)

    def forward(self, x: torch.Tensor):
        x = F.relu(x)
        x = self._conv(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x


class CNNCascadingHead(CNNHead):
    def __init__(self, in_size: int, out_size: int, cascading: bool = True, pool_size: int = 4,
                 layer_norm: bool = True, detach: bool = True):
        super().__init__()
        self._out_size = out_size
        self._pooling = SDNPool(pool_size)
        self._cascading = cascading
        self._detach = detach
        self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
        if self._cascading:
            self._fc = nn.Linear(in_size * pool_size ** 2 + out_size, out_size)
        else:
            self._fc = nn.Linear(in_size * pool_size ** 2, out_size)

    def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
        x = F.relu(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        if self._cascading:
            assert isinstance(cascading_input, torch.Tensor)
            if self._detach:
                cascading_input = cascading_input.detach()
            # apply layer norm to previous logits (from ZTW appendix)
            cascading_input = self.cascading_norm(cascading_input)
            x = torch.cat((x, cascading_input), dim=-1)
        x = self._fc(x)
        return x


class ConvCascadingHead(CNNHead):
    def __init__(self, in_size: int, out_size: int, cascading: bool = True, pool_size: int = 4,
                 layer_norm: bool = True, detach: bool = True):
        super().__init__()
        self._out_size = out_size
        channels = in_size // 2
        self._conv = nn.Conv2d(in_size, channels, kernel_size=3, stride=2)
        self._pooling = SDNPool(pool_size)
        self._cascading = cascading
        self._detach = detach
        self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
        if self._cascading:
            self._fc = nn.Linear(channels * pool_size ** 2 + out_size, out_size)
        else:
            self._fc = nn.Linear(channels * pool_size ** 2, out_size)

    def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
        x = F.relu(x)
        x = self._conv(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        if self._cascading:
            assert isinstance(cascading_input, torch.Tensor)
            if self._detach:
                cascading_input = cascading_input.detach()
            # apply layer norm to previous logits (from ZTW appendix)
            cascading_input = self._cascading_norm(cascading_input)
            x = torch.cat((x, cascading_input), dim=-1)
        x = self._fc(x)
        return x


class TransformerHead(nn.Module):
    """
        Base class for all Transformer heads, allows to check if given head
        is a transformer head or not.
    """

    def __init__(self):
        super().__init__()


class TransformerStandardHead(TransformerHead):
    """ 
        Head that takes only the CLS token from the sequence and then processes it.
    """

    # three arguments for the compatibility with other heads and gpf.py implementation
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self._fc = torch.nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor):
        # take the hidden state corresponding to the first (class) token
        x = x[:, 0]
        x = self._fc(x)
        return x


class TransformerCascadingHead(TransformerHead):
    """
        Cascading (ZTW) head for Transformers.
    """

    def __init__(self, in_size: int, out_size: int, cascading: bool = True, layer_norm: bool = True,
                 detach: bool = True):
        super().__init__()
        self._out_size = out_size
        self._cascading = cascading
        self._detach = detach
        self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
        if self._cascading:
            self._fc = nn.Linear(in_size + out_size, out_size)
        else:
            self._fc = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
        x = x[:, 0]
        if self._cascading:
            assert isinstance(cascading_input, torch.Tensor)
            if self._detach:
                cascading_input = cascading_input.detach()
            # apply layer norm to previous logits (from ZTW paper's appendix)
            cascading_input = self._cascading_norm(cascading_input)
            x = torch.cat((x, cascading_input), dim=-1)
        x = self._fc(x)
        return x


HEAD_TYPES = {
    'cnn_standard_head': CNNStandardHead,
    'cnn_cascading_head': CNNCascadingHead,
    'conv': ConvHead,
    'conv_cascading': ConvCascadingHead,
    'transformer_standard_head': TransformerStandardHead,
    'transformer_cascading_head': TransformerCascadingHead
}

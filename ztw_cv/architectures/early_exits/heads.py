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


def conv_head_class(channel_divisors=(2,), strides=(2,), kernel_sizes=(3,)):
    assert len(channel_divisors) == len(strides) == len(kernel_sizes)

    class ConvHead(CNNHead):
        def __init__(self, in_size: int, out_size: int, pool_size: int = 4):
            super().__init__()
            self._out_size = out_size
            self._convs = nn.ModuleList()
            in_channels = in_size
            for channel_divisor, kernel_size, stride in zip(channel_divisors, kernel_sizes, strides):
                out_channels = in_channels // channel_divisor
                self._convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
                in_channels = out_channels
            self._pooling = SDNPool(pool_size)
            self._fc = nn.Linear(in_channels * pool_size ** 2, out_size)

        def forward(self, x: torch.Tensor):
            for conv in self._convs:
                x = conv(F.relu(x))
            x = self._pooling(x)
            x = x.flatten(1)
            x = self._fc(x)
            return x

    return ConvHead


def conv_cascading_head_class(channel_divisors=(2,), strides=(2,), kernel_sizes=(3,)):
    assert len(channel_divisors) == len(strides) == len(kernel_sizes)

    class ConvCascadingHead(CNNHead):
        def __init__(self, in_size: int, out_size: int, cascading: bool = True, pool_size: int = 4,
                     layer_norm: bool = True, detach: bool = True):
            super().__init__()
            self._out_size = out_size
            self._convs = nn.ModuleList()
            in_channels = in_size
            for channel_divisor, kernel_size, stride in zip(channel_divisors, kernel_sizes, strides):
                out_channels = in_channels // channel_divisor
                self._convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
                in_channels = out_channels
            self._pooling = SDNPool(pool_size)
            self._cascading = cascading
            self._detach = detach
            self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
            if self._cascading:
                self._fc = nn.Linear(in_channels * pool_size ** 2 + out_size, out_size)
            else:
                self._fc = nn.Linear(in_channels * pool_size ** 2, out_size)

        def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
            for conv in self._convs:
                x = conv(F.relu(x))
            x = self._pooling(x)
            x = x.flatten(1)
            if self._cascading:
                assert isinstance(cascading_input, torch.Tensor)
                if self._detach:
                    cascading_input = cascading_input.detach()
                # apply layer norm to previous logits (from ZTW appendix)
                cascading_input = self._cascading_norm(cascading_input)
                x = torch.cat((x, cascading_input), dim=-1)
            x = self._fc(x)
            return x

    return ConvCascadingHead


class ViTHead(nn.Module):

    def __init__(self):
        super().__init__()


def vit_standard_head_class(num_layers, hidden_dim=None):
    assert num_layers > 0

    class ViTStandardHead(ViTHead):
        def __init__(self, in_size: int, out_size: int):
            super().__init__()
            nonlocal hidden_dim
            self._hidden_dim = in_size if hidden_dim is None else hidden_dim
            self._fcs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self._fcs.append(torch.nn.Linear(in_size, self._hidden_dim))
                in_size = self._hidden_dim
            self._fcs.append(torch.nn.Linear(in_size, out_size))

        def forward(self, x: torch.Tensor):
            # take the hidden state corresponding to the first (class) token
            x = x[:, 0]
            for i in range(num_layers - 1):
                x = F.gelu(self._fcs[i](x))
            x = self._fcs[-1](x)
            return x

    return ViTStandardHead


def vit_cascading_head_class(num_layers, hidden_dim=None):
    assert num_layers > 0

    class ViTCascadingHead(ViTHead):
        def __init__(self, in_size: int, out_size: int, cascading: bool = True, layer_norm: bool = True,
                     detach: bool = True):
            super().__init__()
            nonlocal hidden_dim
            self._hidden_dim = in_size if hidden_dim is None else hidden_dim
            self._out_size = out_size
            self._cascading = cascading
            self._detach = detach
            self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
            self._fcs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self._fcs.append(torch.nn.Linear(in_size, self._hidden_dim))
                in_size = self._hidden_dim
            if self._cascading:
                self._fcs.append(torch.nn.Linear(in_size + out_size, out_size))
            else:
                self._fcs.append(torch.nn.Linear(in_size, out_size))

        def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
            # take the hidden state corresponding to the first (class) token
            x = x[:, 0]
            for i in range(num_layers - 1):
                x = F.gelu(self._fcs[i](x))
            if self._cascading:
                assert isinstance(cascading_input, torch.Tensor)
                if self._detach:
                    cascading_input = cascading_input.detach()
                # apply layer norm to previous logits (from ZTW paper's appendix)
                cascading_input = self._cascading_norm(cascading_input)
                x = torch.cat((x, cascading_input), dim=-1)
            x = self._fcs[-1](x)
            return x

    return ViTCascadingHead


class SwinHead(nn.Module):

    def __init__(self):
        super().__init__()


def swin_standard_head_class(num_layers, hidden_dim=None):
    assert num_layers > 0

    class SwinStandardHead(SwinHead):
        def __init__(self, in_size: int, out_size: int):
            super().__init__()
            nonlocal hidden_dim
            self._hidden_dim = in_size if hidden_dim is None else hidden_dim
            self._pooling = SDNPool(1)
            self._fcs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self._fcs.append(torch.nn.Linear(in_size, self._hidden_dim))
                in_size = self._hidden_dim
            self._fcs.append(torch.nn.Linear(in_size, out_size))

        def forward(self, x: torch.Tensor):
            x = x.permute([0, 3, 2, 1])
            x = self._pooling(x)
            x = x.flatten(1)
            for i in range(num_layers - 1):
                x = F.gelu(self._fcs[i](x))
            x = self._fcs[-1](x)
            return x

    return SwinStandardHead


def swin_cascading_head_class(num_layers, hidden_dim=None):
    assert num_layers > 0

    class SwinCascadingHead(SwinHead):
        def __init__(self, in_size: int, out_size: int, cascading: bool = True, layer_norm: bool = True,
                     detach: bool = True):
            super().__init__()
            nonlocal hidden_dim
            self._hidden_dim = in_size if hidden_dim is None else hidden_dim
            self._pooling = SDNPool(1)
            self._out_size = out_size
            self._cascading = cascading
            self._detach = detach
            self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
            self._fcs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self._fcs.append(torch.nn.Linear(in_size, self._hidden_dim))
                in_size = self._hidden_dim
            if self._cascading:
                self._fcs.append(torch.nn.Linear(in_size + out_size, out_size))
            else:
                self._fcs.append(torch.nn.Linear(in_size, out_size))

        def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
            x = x.permute([0, 3, 2, 1])
            x = self._pooling(x)
            x = x.flatten(1)
            for i in range(num_layers - 1):
                x = F.gelu(self._fcs[i](x))
            if self._cascading:
                assert isinstance(cascading_input, torch.Tensor)
                if self._detach:
                    cascading_input = cascading_input.detach()
                # apply layer norm to previous logits (from ZTW paper's appendix)
                cascading_input = self._cascading_norm(cascading_input)
                x = torch.cat((x, cascading_input), dim=-1)
            x = self._fcs[-1](x)
            return x

    return SwinCascadingHead


HEAD_TYPES = {
    'pooled_head': conv_head_class((), (), ()),
    'pooled_cascading_head': conv_cascading_head_class((), (), ()),
    'conv': conv_head_class((2,), (2,), (3,)),
    'conv_cascading': conv_cascading_head_class((2,), (2,), (3,)),
    'conv_2l': conv_head_class((1, 2), (2, 1), (3, 3)),
    'conv_cascading_2l': conv_cascading_head_class((1, 2), (2, 1), (3, 3)),
    'conv_3l': conv_head_class((1, 1, 2), (2, 1, 1), (3, 3, 3)),
    'conv_cascading_3l': conv_cascading_head_class((1, 1, 2), (2, 1, 1), (3, 3, 3)),
    'conv_4l': conv_head_class((2, 1, 1, 2), (1, 1, 2, 1), (3, 3, 3, 3)),
    'conv_cascading_4l': conv_cascading_head_class((2, 1, 1, 2), (1, 1, 2, 1), (3, 3, 3, 3)),
    'conv_5l': conv_head_class((2, 1, 1, 1, 1), (1, 1, 1, 1, 1), (3, 3, 3, 3, 3)),
    'conv_cascading_5l': conv_cascading_head_class((2, 1, 1, 1, 1), (1, 1, 1, 1, 1), (3, 3, 3, 3, 3)),
    'vit_standard_head': vit_standard_head_class(1),
    'vit_2l_head': vit_standard_head_class(2, 1024),
    'vit_2l_2048_head': vit_standard_head_class(2, 2048),
    'vit_2l_4096_head': vit_standard_head_class(2, 4096),
    'vit_2l_8192_head': vit_standard_head_class(2, 8192),
    'vit_3l_head': vit_standard_head_class(3, 8192),
    'vit_4l_head': vit_standard_head_class(4, 8192),
    'vit_5l_head': vit_standard_head_class(5, 8192),
    'vit_cascading_head': vit_cascading_head_class(1),
    'vit_cascading_2l_head': vit_cascading_head_class(2, 1024),
    'vit_cascading_2l_2048_head': vit_cascading_head_class(2, 2048),
    'vit_cascading_2l_4096_head': vit_cascading_head_class(2, 4096),
    'vit_cascading_2l_8192_head': vit_cascading_head_class(2, 8192),
    'vit_cascading_3l_head': vit_cascading_head_class(3, 8192),
    'vit_cascading_4l_head': vit_cascading_head_class(4, 8192),
    'vit_cascading_5l_head': vit_cascading_head_class(5, 8192),
    'swin_standard_head': swin_standard_head_class(1),
    'swin_2l_head': swin_standard_head_class(2, 1024),
    'swin_cascading_head': swin_cascading_head_class(1),
    'swin_cascading_2l_head': swin_cascading_head_class(2, 1024),
}

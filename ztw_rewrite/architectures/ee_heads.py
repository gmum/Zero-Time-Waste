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
        avg_p = self._alpha * self._max_pool(x)
        max_p = (1 - self._alpha) * self._avg_pool(x)
        mixed = avg_p + max_p
        return mixed


class StandardHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 4):
        super().__init__()
        self._num_classes = num_classes
        self._pooling = SDNPool(pool_size)
        self._fc = nn.Linear(in_channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 4):
        super().__init__()
        self._num_classes = num_classes
        channels = in_channels // 2
        self._conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2)
        self._pooling = SDNPool(pool_size)
        self._fc = nn.Linear(channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(x)
        x = self._conv(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x


class TooBigConvHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 4):
        super().__init__()
        self._num_classes = num_classes
        channels = in_channels // 2
        self._conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding='same')
        self._conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same')
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same')
        self._pooling = SDNPool(pool_size)
        self._fc = nn.Linear(channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(x)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x


class StandardCascadingHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, cascading: bool = True, pool_size: int = 4,
                 layer_norm: bool = True, detach: bool = True):
        super().__init__()
        self._num_classes = num_classes
        self._pooling = SDNPool(pool_size)
        self._cascading = cascading
        self._detach = detach
        if layer_norm:
            self.cascading_norm = nn.LayerNorm(num_classes)
        else:
            self.cascading_norm = nn.Identity()
        if self._cascading:
            self._fc = nn.Linear(in_channels * pool_size ** 2 + num_classes, num_classes)
        else:
            self._fc = nn.Linear(in_channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
        x = F.relu(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        if self._cascading:
            assert isinstance(cascading_input, torch.Tensor)
            if self._detach:
                cascading_input = cascading_input.detach()
            x = torch.cat((x, cascading_input), dim=-1)
        x = self._fc(x)
        return x


class ConvCascadingHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, cascading: bool = True, pool_size: int = 4,
                 layer_norm: bool = True, detach: bool = True):
        super().__init__()
        self._num_classes = num_classes
        channels = in_channels // 2
        self._conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2)
        self._pooling = SDNPool(pool_size)
        self._cascading = cascading
        self._detach = detach
        if layer_norm:
            self.cascading_norm = nn.LayerNorm(num_classes)
        else:
            self.cascading_norm = nn.Identity()
        if self._cascading:
            self._fc = nn.Linear(channels * pool_size ** 2 + num_classes, num_classes)
        else:
            self._fc = nn.Linear(channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
        x = F.relu(x)
        x = self._conv(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        if self._cascading:
            assert isinstance(cascading_input, torch.Tensor)
            if self._detach:
                cascading_input = cascading_input.detach()
            x = torch.cat((x, cascading_input), dim=-1)
        x = self._fc(x)
        return x

class TooBigConvCascadingHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, cascading: bool = True, pool_size: int = 4,
                 layer_norm: bool = True, detach: bool = True):
        super().__init__()
        self._num_classes = num_classes
        channels = in_channels // 2
        self._conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding='same')
        self._conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same')
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same')
        self._pooling = SDNPool(pool_size)
        self._cascading = cascading
        self._detach = detach
        if layer_norm:
            self.cascading_norm = nn.LayerNorm(num_classes)
        else:
            self.cascading_norm = nn.Identity()
        if self._cascading:
            self._fc = nn.Linear(channels * pool_size ** 2 + num_classes, num_classes)
        else:
            self._fc = nn.Linear(channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
        x = F.relu(x)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        if self._cascading:
            assert isinstance(cascading_input, torch.Tensor)
            if self._detach:
                cascading_input = cascading_input.detach()
            x = torch.cat((x, cascading_input), dim=-1)
        x = self._fc(x)
        return x


HEAD_TYPES = {
    'standard': StandardHead,
    'cascading': StandardCascadingHead,
    'conv': ConvHead,
    'conv_cascading': ConvCascadingHead,
    'too_big': TooBigConvHead,
    'too_big_cascading': TooBigConvCascadingHead,
}

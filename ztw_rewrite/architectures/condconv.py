import math
from typing import List
from typing import Tuple, Any, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = \
        num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w


class CondConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = 1,
                 padding: Union[str, int] = 0,
                 dilation: Tuple[int, ...] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Any = None,
                 dtype: Any = None):
        assert groups == 1, f'The layer does not support groups yet'
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
        )
        filter_in_channels = torch.linspace(1.0, float(in_channels), steps=out_channels).round().to(torch.int)
        c_out, c_in, kh, kw = self.weight.shape
        mask = torch.zeros(c_out, c_in, dtype=dtype, device=device)
        for i, in_channels_num in enumerate(filter_in_channels):
            mask[i, :in_channels_num] = 1.
        mask = mask.view(c_out, c_in, 1, 1)
        self.register_buffer('mask', mask)
        self._k = self.in_channels
        self._mode = 'standard'

    def set_k(self, k):
        assert k > 0
        self._k = k

    def execution_mode(self, mode):
        assert mode in ['standard', 'benchmark']
        self._mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if self._mode == 'standard':
            weight = self.weight * self.mask
            weight[self._k:] = 0.
            if self.bias is None:
                return self._conv_forward(x, weight, None)
            else:
                bias = self.bias
                bias[self._k:] = 0.
                return self._conv_forward(x, weight, bias, )
        elif self._mode == 'benchmark':
            h_in, w_in = x.size(2), x.size(3)
            h_out, w_out = conv2d_output_shape((h_in, w_in), self.kernel_size, self.stride, self.padding, self.dilation)
            output = torch.empty(x.size(0), self.out_channels, h_out, w_out,
                                 dtype=x.dtype, device=x.device)
            for filter_idx in range(self._k):
                in_channels = self.mask[filter_idx].sum().to(torch.int).item()
                weight = self.weight[filter_idx:filter_idx + 1, :in_channels]
                bias = self.bias[filter_idx:filter_idx + 1] if self.bias is not None else None
                output[:, filter_idx:filter_idx + 1] = F.conv2d(x[:, :in_channels], weight, bias, self.stride,
                                                                self.padding, self.dilation, self.groups)
            # we need to fill the rest with zeroes
            output[:, self._k:] = 0
            return output


class CondBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = CondConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CondConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CondConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CondBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = CondConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CondConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = CondConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CondConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CondResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 10, extension: int = 1) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.in_planes = 64 * extension
        num_channels = 64 * extension
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.layer1 = self._make_layer(block, num_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * num_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * num_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * num_channels, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * num_channels * block.expansion, num_classes)
        self._k_fraction = 1.0
        self._fc_k_fractions = [1.0]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @property
    def number_of_classes(self):
        return self._num_classes

    def _set_modules_k_fraction(self, k_fraction):
        for m in self.modules():
            if isinstance(m, CondConv2d):
                k = int(round(m.out_channels * k_fraction))
                m.set_k(k)

    def set_k_fraction(self, k_fraction):
        self._k_fraction = k_fraction
        # assumes we always use 1.0 for training
        # self._set_modules_k_fraction(1.0)

    def set_fc_k_fractions(self, k_fractions):
        self._fc_k_fractions = k_fractions

    def train(self, mode: bool = True):
        if mode:
            self._set_modules_k_fraction(1.0)
        else:
            self._set_modules_k_fraction(self._k_fraction)
        nn.Module.train(self, mode)

    def eval(self):
        self.train(False)

    def execution_mode(self, mode: str):
        for m in self.modules():
            if isinstance(m, CondConv2d):
                m.execution_mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = {}
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        full_k = self.linear.out_features
        if self.training:
            # TODO add other variants for the linear layer
            # such as: with masks indicating whether the value is missing, multiple heads etc.
            for k_fraction in self._fc_k_fractions:
                k = int(round(full_k * k_fraction))
                linear_weights = self.linear.weight[:, k:]
                outputs[k] = F.linear(out[:, k:], linear_weights, bias=self.linear.bias)
            return outputs
        else:
            k = int(round(full_k * self._k_fraction))
            linear_weights = self.linear.weight[:, k:]
            out = F.linear(out[:, k:], linear_weights, bias=self.linear.bias)
            return out


def CondResNet18(num_classes: int = 10):
    return CondResNet(CondBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def CondResNet34(num_classes: int = 10):
    return CondResNet(CondBasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def CondResNet50(num_classes: int = 10):
    return CondResNet(CondBottleneck, [3, 4, 6, 3], num_classes=num_classes)


def CondResNet101(num_classes: int = 10):
    return CondResNet(CondBottleneck, [3, 4, 23, 3], num_classes=num_classes)


def CondResNet152(num_classes: int = 10):
    return CondResNet(CondBottleneck, [3, 8, 36, 3], num_classes=num_classes)

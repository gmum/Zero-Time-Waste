from itertools import chain
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


# TODO perhaps use the torchvision model instead (note the difference in the initial convolution)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 10, extension: int = 1,
                 standard_first_conv=True):
        super().__init__()
        self.input_channels = 3
        self._num_classes = num_classes
        self.in_planes = 64 * extension
        num_channels = 64 * extension
        if standard_first_conv is True:
            self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.layer1 = self._make_layer(block, num_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * num_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * num_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * num_channels, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * num_channels * block.expansion, num_classes)

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

    def forward_generator(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        for block in chain(self.layer1, self.layer2, self.layer3, self.layer4):
            x = block(x)
            x = yield x, None
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        y = self.linear(x)
        _ = yield None, y

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def ResNet18(num_classes: int = 1000, standard_first_conv=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, standard_first_conv=standard_first_conv)


def ResNet34(num_classes: int = 1000, standard_first_conv=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, standard_first_conv=standard_first_conv)


def ResNet50(num_classes: int = 1000, standard_first_conv=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, standard_first_conv=standard_first_conv)


def ResNet101(num_classes: int = 1000, standard_first_conv=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, standard_first_conv=standard_first_conv)


def ResNet152(num_classes: int = 1000, standard_first_conv=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, standard_first_conv=standard_first_conv)

from itertools import chain

from torch import nn

# TODO perhaps use the torchvision model instead (all these models seems to differ slightly)


class wide_basic(nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, stride=1):
        super().__init__()
        self.layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True))
        conv_layer.append(nn.Dropout(p=dropout_rate))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True))
        self.layers.append(nn.Sequential(*conv_layer))
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            shortcut = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True), )
        self.layers.append(shortcut)

    def forward(self, x):
        out = self.layers[0](x)
        out += self.layers[1](x)
        return out


class WideResNet(nn.Module):
    def __init__(self, num_blocks, widen_factor, num_classes, dropout_rate, standard_first_conv=True):
        super().__init__()
        self.input_channels = 3
        self.num_blocks = num_blocks
        self.widen_factor = widen_factor
        self._num_classes = num_classes
        self.dropout_rate = dropout_rate
        if standard_first_conv is True:
            self.initial_conv = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=True)
        else:
            self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.layers = nn.ModuleList()
        self.in_channels = 16
        self.layers.extend(self._wide_layer(wide_basic, self.in_channels * self.widen_factor, block_id=0, stride=1))
        self.layers.extend(self._wide_layer(wide_basic, 32 * self.widen_factor, block_id=1, stride=2))
        self.layers.extend(self._wide_layer(wide_basic, 64 * self.widen_factor, block_id=2, stride=2))
        final_layers = []
        final_layers.append(nn.BatchNorm2d(64 * self.widen_factor, momentum=0.9))
        final_layers.append(nn.ReLU(inplace=True))
        final_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        final_layers.append(nn.Flatten())
        final_layers.append(nn.Linear(64 * self.widen_factor, self._num_classes))
        self.final_layers = nn.Sequential(*final_layers)

    def _wide_layer(self, block, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, self.dropout_rate, stride))
            self.in_channels = channels
        return layers

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward_generator(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
            x = yield x, None
        x = self.final_layers(x)
        _ = yield None, x

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layers(x)
        return x

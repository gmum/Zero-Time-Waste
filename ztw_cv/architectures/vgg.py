from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool_size, batch_norm):
        super(ConvBlock, self).__init__()
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                     padding=1))
        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())
        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))
        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class FcBlock(nn.Module):
    def __init__(self, in_size, out_size, flatten=True):
        super(FcBlock, self).__init__()
        fc_layers = []
        if flatten:
            fc_layers.append(nn.Flatten())
        fc_layers.append(nn.Linear(in_size, out_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class VGG(nn.Module):
    def __init__(self, input_size, conv_channels, conv_bn, max_pool_sizes, fc_sizes, num_classes):
        super().__init__()
        self.input_size = input_size
        self.input_channels = 3
        self._conv_channels = conv_channels
        self._conv_bn = conv_bn
        self._max_pool_sizes = max_pool_sizes
        self._fc_layer_sizes = fc_sizes
        self._num_classes = num_classes
        #
        self.layers = nn.ModuleList()
        # add conv layers
        input_channels = self.input_channels
        cur_input_size = self.input_size
        for channels, max_pool_size in zip(self._conv_channels, self._max_pool_sizes):
            assert max_pool_size in {1, 2}
            if max_pool_size == 2:
                cur_input_size = int(cur_input_size / 2)
            self.layers.append(ConvBlock(input_channels, channels, max_pool_size, self._conv_bn))
            input_channels = channels
        # add fc layers
        fc_input_size = cur_input_size * cur_input_size * self._conv_channels[-1]
        for layer_id, width in enumerate(self._fc_layer_sizes[:-1]):
            flatten = False
            if layer_id == 0:
                flatten = True
            self.layers.append(FcBlock(fc_input_size, width, flatten=flatten))
            fc_input_size = width
        final_layers = []
        final_layers.append(nn.Linear(fc_input_size, self._fc_layer_sizes[-1]))
        final_layers.append(nn.Dropout(0.5))
        final_layers.append(nn.Linear(self._fc_layer_sizes[-1], self._num_classes))
        self.final_layers = nn.Sequential(*final_layers)

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward_generator(self, x):
        assert x.size(2) == self.input_size and x.size(3) == self.input_size, f'{x.size()=}'
        for layer in self.layers:
            x = layer(x)
            x = yield x, None
        x = self.final_layers(x)
        _ = yield None, x

    def forward(self, x):
        assert x.size(2) == self.input_size and x.size(3) == self.input_size, f'{x.size()=}'
        for layer in self.layers:
            x = layer(x)
        x = self.final_layers(x)
        return x


def VGG16BN(input_size, num_classes):
    conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    if input_size == 32:
        fc_sizes = [512, 512]
    else:
        fc_sizes = [2048, 1024]
    max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    return VGG(input_size, conv_channels, conv_bn=True, max_pool_sizes=max_pool_sizes, fc_sizes=fc_sizes,
               num_classes=num_classes)

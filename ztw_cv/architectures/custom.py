import torch
from torch import nn


class FCNet(nn.Module):
    def __init__(self, input_size, channels, num_layers, layer_size, classes):
        super().__init__()
        assert input_size > 1
        assert channels >= 1
        assert num_layers > 1
        assert layer_size > 1
        assert classes > 1
        self.input_size = input_size
        self.input_channels = channels
        self.num_layers = num_layers
        self.layer_size = layer_size
        self._num_classes = classes
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(nn.Linear(self.input_size ** 2 * self.input_channels, self.layer_size))
        num_layers -= 1
        # remaining layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.layer_size, self.layer_size))
        self.layers.append(nn.Linear(self.layer_size, self._num_classes))

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc_layer in self.layers:
            x = torch.relu(fc_layer(x))
        return x

    def forward_generator(self, x):
        x = x.view(x.size(0), -1)
        for fc_layer in self.layers[:-1]:
            x = torch.relu(fc_layer(x))
            x = yield x, None
        x = torch.relu(self.layers[-1](x))
        yield None, x


class DCNet(nn.Module):
    def __init__(self, input_size, channels, num_layers, num_filters, kernel_size, classes, batchnorm=True):
        super().__init__()
        assert input_size > 1
        assert channels >= 1
        assert classes > 1
        assert num_layers >= 1
        self.input_size = input_size
        self.input_channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self._num_classes = classes
        self.batchnorm = batchnorm
        self.layers = nn.ModuleList()
        if self.batchnorm:
            self.bn_layers = nn.ModuleList()
        # assume, for simplicity, that we only use 'same' padding and stride 1
        padding = (self.kernel_size - 1) // 2
        c_in = self.input_channels
        c_out = self.num_filters
        for layer in range(self.num_layers):
            self.layers.append(nn.Conv2d(c_in, c_out, kernel_size=self.kernel_size, stride=1, padding=padding))
            c_in, c_out = c_out, c_out
            # c_in, c_out = c_out, c_out + self.filters_inc
            if self.batchnorm:
                self.bn_layers.append(nn.BatchNorm2d(c_out))
        self.layers.append(nn.Linear(c_out, self._num_classes))

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            if self.batchnorm:
                x = self.bn_layers[i](x)
        x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
        last_activations = self.layers[-1](x_transformed)
        return last_activations

    def forward_generator(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            if self.batchnorm:
                x = self.bn_layers[i](x)
            x = yield x, None
        x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
        last_activations = self.layers[-1](x_transformed)
        _ = yield None, last_activations

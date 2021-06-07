import math

import aux_funcs as af
import model_funcs as mf
import numpy as np
import torch
import torch.nn as nn


class wide_basic(nn.Module):
    def __init__(self, args, in_channels, channels, dropout_rate, params, head_variant, heads_per_ensemble, stride=1, prev_dim=0):
        super(wide_basic, self).__init__()

        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.head_variant = head_variant
        self.heads_per_ensemble = heads_per_ensemble

        self.depth = 2

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

        if add_output:
            self.output = af.InternalClassifier(args,
                                                input_size,
                                                channels,
                                                num_classes,
                                                head_variant=self.head_variant,
                                                heads_per_ensemble=self.heads_per_ensemble,
                                                prev_dim=prev_dim)
            self.no_output = False
        else:
            self.output = None
            self.forward = self.only_forward
            self.no_output = True

    def only_output(self, x, prev_output=None):
        fwd = self.layers[0](x)
        fwd = fwd + self.layers[1](x)
        out = self.output(fwd, prev_output)  # output layers for this module
        return out

    def only_forward(self, x, prev_output=None):
        fwd = self.layers[0](x)
        fwd = fwd + self.layers[1](x)
        return fwd, 0, None

    def forward(self, x, prev_output=None):
        fwd = self.layers[0](x)
        fwd = fwd + self.layers[1](x)
        return fwd, 1, self.output(fwd, prev_output)


class WideResNet_SDN(nn.Module):
    def __init__(self, args, params):
        super(WideResNet_SDN, self).__init__()
        self.args = args
        self.num_blocks = params['num_blocks']
        self.widen_factor = params['widen_factor']
        self.num_classes = int(params['num_classes'])
        self.dropout_rate = params['dropout_rate']
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.add_out_nonflat = params['add_ic']
        self.add_output = [item for sublist in self.add_out_nonflat for item in sublist]
        self.init_weights = params['init_weights']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.in_channels = 16
        self.num_output = sum(self.add_output) + 1
        self.head_variant = params['head_variant']
        self.heads_per_ensemble = params['heads_per_ensemble']
        self.first_internal = True

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        if self.input_size == 32:  # cifar10 and cifar100
            self.cur_input_size = self.input_size
            self.init_conv = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        elif self.input_size == 64:  # tiny imagenet
            self.cur_input_size = int(self.input_size / 2)
            self.init_conv = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.layers = nn.ModuleList()
        self.layers.extend(
            self._wide_layer(args,
                             wide_basic,
                             self.in_channels * self.widen_factor,
                             block_id=0,
                             head_variant=self.head_variant,
                             heads_per_ensemble=self.heads_per_ensemble,
                             stride=1))
        self.cur_input_size = int(self.cur_input_size / 2)
        self.layers.extend(self._wide_layer(args, 
                                            wide_basic,
                                            32 * self.widen_factor, 
                                            block_id=1,
                                            head_variant=self.head_variant,
                                            heads_per_ensemble=self.heads_per_ensemble,
                                            stride=2))
        self.cur_input_size = int(self.cur_input_size / 2)
        self.layers.extend(self._wide_layer(args, 
                                            wide_basic,
                                            64 * self.widen_factor,
                                            block_id=2,
                                            head_variant=self.head_variant,
                                            heads_per_ensemble=self.heads_per_ensemble,
                                            stride=2))

        end_layers = []

        end_layers.append(nn.BatchNorm2d(64 * self.widen_factor))
        end_layers.append(nn.ReLU(inplace=True))
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(af.Flatten())
        end_layers.append(nn.Linear(64 * self.widen_factor, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def _wide_layer(self, args, block, channels, block_id, head_variant, heads_per_ensemble, stride):
        num_blocks = self.num_blocks[block_id]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for cur_block_id, stride in enumerate(strides):
            add_output = self.add_out_nonflat[block_id][cur_block_id]
            params = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            prev_dim = 0 if self.first_internal or not self.args.stacking else self.num_classes
            layers.append(block(args, self.in_channels, channels, self.dropout_rate, params, head_variant, heads_per_ensemble, stride, prev_dim=prev_dim))
            self.in_channels = channels
            self.cur_output_id += add_output
            if add_output:
                self.first_internal = False

        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        prev_output = None
        for layer in self.layers:
            fwd, is_output, output = layer(fwd, prev_output if self.args.stacking else None)
            if is_output:
                outputs.append(output)
                prev_output = output
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit(self, x):
        confidences = []
        outputs = []
        prev_output = None

        fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd, prev_output if self.args.stacking else None)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)

                confidence = torch.max(softmax)
                confidences.append(confidence)

                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early

                output_id += is_output
                prev_output = output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early

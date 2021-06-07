import math

import aux_funcs as af
import model_funcs as mf
import numpy as np
import torch
import torch.nn as nn


class ConvBlockWOutput(nn.Module):
    def __init__(self, args, conv_params, output_params, head_variant, heads_per_ensemble, prev_dim=0):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.head_variant = head_variant
        self.heads_per_ensemble = heads_per_ensemble

        self.depth = 1

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))

        self.layers = nn.Sequential(*conv_layers)

        if add_output:
            self.output = af.InternalClassifier(args,
                                                input_size,
                                                output_channels,
                                                num_classes,
                                                head_variant=self.head_variant,
                                                heads_per_ensemble=self.heads_per_ensemble,
                                                prev_dim=prev_dim)
            self.no_output = False

        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x, prev_output=None):
        fwd = self.layers(x)
        output = self.output(fwd, prev_output)
        return fwd, 1, output

    def only_output(self, x, prev_output=None):
        fwd = self.layers(x)
        out = self.output(fwd, prev_output)  # output layers for this module
        return out

    def only_forward(self, x, prev_output=None):
        fwd = self.layers(x)
        return fwd, 0, None


class FcBlockWOutput(nn.Module):
    def __init__(self, args, fc_params, output_params, flatten=False, prev_dim=0):
        super(FcBlockWOutput, self).__init__()
        input_size = fc_params[0]
        output_size = fc_params[1]

        add_output = output_params[0]
        num_classes = output_params[1]
        self.output_id = output_params[2]
        self.depth = 1

        fc_layers = []

        if flatten:
            fc_layers.append(af.Flatten())

        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = nn.Linear(output_size + prev_dim, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x, prev_output=None):
        fwd = self.layers(x)
        
        if prev_output is not None:
            out = torch.cat([fwd, prev_output[0]], -1)
        else:
            out = fwd
        output = self.output(out)
        return fwd, 1, [output]

    def only_output(self, x, prev_output=None):
        fwd = self.layers(x)
        if prev_output is not None:
            out = torch.cat([fwd, prev_output[0]], -1)
        else:
            out = fwd
        return [self.output(out)]

    def only_forward(self, x, prev_output=None):
        return self.layers(x), 0, None


class VGG_SDN(nn.Module):
    def __init__(self, args, params):
        super(VGG_SDN, self).__init__()
        self.args = args
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels']  # the first element is input dimension
        self.fc_layer_sizes = params['fc_layers']

        # read or assign defaults to the rest
        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.add_output = params['add_ic']

        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1
        self.head_variant = params['head_variant']
        self.heads_per_ensemble = params['heads_per_ensemble']

        self.init_conv = nn.Sequential()  # just for compatibility with other models
        self.layers = nn.ModuleList()
        self.init_depth = 0
        self.end_depth = 2

        self.first_internal = True

        # add conv layers
        input_channel = 3
        cur_input_size = self.input_size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)

            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)

            prev_dim = 0 if self.first_internal or not self.args.stacking else self.num_classes
            self.layers.append(
                ConvBlockWOutput(args,
                                 conv_params,
                                 output_params,
                                 head_variant=self.head_variant,
                                 heads_per_ensemble=self.heads_per_ensemble,
                                 prev_dim=prev_dim))
            if add_output:
                self.first_internal = False
            input_channel = channel
            output_id += add_output

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            add_output = self.add_output[layer_id + len(self.conv_channels)]
            output_params = (add_output, self.num_classes, output_id)
            prev_dim = 0 if self.first_internal or not self.args.stacking else self.num_classes
            self.layers.append(FcBlockWOutput(args, fc_params, output_params, flatten=flatten, prev_dim=prev_dim))
            if add_output:
                self.first_internal = False
            fc_input_size = width
            output_id += add_output

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
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

        fwd = self.init_conv(x)
        prev_output = None
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd, prev_output if self.args.stacking else None)

            if is_output:
                outputs.append(output)
                prev_output = output
                softmax = nn.functional.softmax(output[0], dim=0)

                confidence = torch.max(softmax)
                confidences.append(confidence)

                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early

                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early

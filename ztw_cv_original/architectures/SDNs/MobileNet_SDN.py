import aux_funcs as af
import model_funcs as mf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockWOutput(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, args, in_channels, out_channels, params, head_variant, heads_per_ensemble, stride=1, prev_dim=0):
        super(BlockWOutput, self).__init__()

        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.head_variant = head_variant
        self.heads_per_ensemble = heads_per_ensemble

        self.depth = 2

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

        if add_output:
            self.output = af.InternalClassifier(args,
                                                input_size,
                                                out_channels,
                                                num_classes,
                                                head_variant=self.head_variant,
                                                heads_per_ensemble=self.heads_per_ensemble,
                                                prev_dim=prev_dim)
            self.no_output = False

        else:
            self.forward = self.only_forward
            self.output = nn.Sequential()
            self.no_output = True

    def forward(self, x, prev_output=None):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd, prev_output)

    def only_output(self, x, prev_output=None):
        fwd = self.layers(x)
        return self.output(fwd, prev_output)

    def only_forward(self, x, prev_output=None):
        return self.layers(x), 0, None


class MobileNet_SDN(nn.Module):
    # (128,2) means conv channels=128, conv stride=2, by default conv stride=1
    def __init__(self, args, params):
        super(MobileNet_SDN, self).__init__()
        self.args = args
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.add_output = params['add_ic']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1
        self.head_variant = params['head_variant']
        self.heads_per_ensemble = params['heads_per_ensemble']
        self.in_channels = 32
        self.cur_input_size = self.input_size
        self.first_internal = True

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        init_conv = []
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []
        if self.input_size == 32:  # cifar10 and cifar100
            end_layers.append(nn.AvgPool2d(2))
        elif self.input_size == 64:  # tiny imagenet
            end_layers.append(nn.AvgPool2d(4))

        end_layers.append(af.Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []

        for block_id, x in enumerate(self.cfg):
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if stride == 2:
                self.cur_input_size = int(self.cur_input_size / 2)

            add_output = self.add_output[block_id]
            params = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            prev_dim = 0 if self.first_internal or not self.args.stacking else self.num_classes
            layers.append(
                BlockWOutput(self.args, in_channels, out_channels, params, self.head_variant, self.heads_per_ensemble,
                             stride, prev_dim=prev_dim))
            in_channels = out_channels
            if add_output:
                self.first_internal = False
            self.cur_output_id += add_output

        return layers

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

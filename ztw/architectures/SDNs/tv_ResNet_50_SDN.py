import torch
from torch import nn
from torchvision import models
from itertools import chain

import aux_funcs as af
import model_funcs as mf


class ResNet50_SDN(nn.Module):
    def __init__(self, args, params=None):
        super().__init__()

        self.args = args

        assert self.args.heads_per_ensemble == 1
        self.core_model = models.resnet50(pretrained=True, progress=True)
        self.num_classes = self.core_model.fc.out_features
        self.train_func = mf.sdn_train
        self.augment_training = params['augment_training']

        self.head_variant = params['head_variant']
        self.heads_per_ensemble = params['heads_per_ensemble']
        self.test_func = mf.sdn_test
        self.first_internal = True

        self.shift = args.head_shift

        self.output_ics = nn.ModuleList()
        #
        x = torch.randn(1, 3, 224, 224)
        x = self.core_model.conv1(x)
        x = self.core_model.bn1(x)
        x = self.core_model.relu(x)
        x = self.core_model.maxpool(x)
        for i, module in enumerate(self.core_bottlenecks()):
            x = module(x)
            if self.args.heads == 'half' and (i - self.shift) % 2:
                continue
            if self.args.heads == 'third' and (i - self.shift) % 3:
                continue
            elif self.args.heads == 'quarter' and (i - self.shift) % 4:
                continue
            prev_dim = 0 if self.first_internal or not self.args.stacking else self.num_classes
            input_size = x.size(2)
            output_channels = module.conv3.out_channels
            self.output_ics.append(
                af.InternalClassifier(args,
                                      input_size,
                                      output_channels,
                                      self.num_classes,
                                      head_variant=self.args.head_arch,
                                      heads_per_ensemble=1,
                                      prev_dim=prev_dim))
            self.first_internal = False

        self.num_output = len(self.output_ics) + 1

    def core_bottlenecks(self):
        return chain(self.core_model.layer1, self.core_model.layer2, self.core_model.layer3, self.core_model.layer4)

    def modules(self):
        # it is necessary to overwrite this as code in profiler.py depends on the order of items in this iterator
        ord_modules = []
        ord_modules.append(self.core_model.conv1)
        ord_modules.append(self.core_model.bn1)
        ord_modules.append(self.core_model.relu)
        ord_modules.append(self.core_model.maxpool)
        for i, module in enumerate(self.core_bottlenecks()):
            ord_modules.append(module)
            if self.args.heads == 'original':
                ind = i
            elif self.args.heads == 'half' and (i - self.shift) % 2 == 0:
                ind = (i - self.shift) // 2
            elif self.args.heads == 'third' and (i - self.shift) % 3 == 0:
                ind = (i - self.shift) // 3
            elif self.args.heads == 'quarter' and (i - self.shift) % 4 == 0:
                ind = (i - self.shift) // 4
            else:
                continue
            ord_modules.append(self.output_ics[ind])
        ord_modules.append(self.core_model.avgpool)
        ord_modules.append(self.core_model.fc)
        return chain.from_iterable(m.modules() for m in ord_modules)

    def forward(self, x):
        outputs = []

        with torch.no_grad():
            x = self.core_model.conv1(x)
            x = self.core_model.bn1(x)
            x = self.core_model.relu(x)
            x = self.core_model.maxpool(x)

        # bottleneck blocks
        prev_output = None
        for i, module in enumerate(self.core_bottlenecks()):
            with torch.no_grad():
                x = module(x)
            if self.args.heads == 'original':
                ind = i
            elif self.args.heads == 'half' and (i - self.shift) % 2 == 0:
                ind = (i - self.shift) // 2
            elif self.args.heads == 'third' and (i - self.shift) % 3 == 0:
                ind = (i - self.shift) // 3
            elif self.args.heads == 'quarter' and (i - self.shift) % 4 == 0:
                ind = (i - self.shift) // 4
            else:
                continue
            output = self.output_ics[ind](x, prev_output=prev_output if self.args.stacking else None)
            outputs.append(output)
            prev_output = output

        with torch.no_grad():
            x = self.core_model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.core_model.fc(x)
            outputs.append(x)
        return outputs

from itertools import chain

from torch import nn
from torchvision import models

import model_funcs as mf


class ResNet50(nn.Module):
    def __init__(self, args, model_params=None):
        super().__init__()

        self.args = args
        self.core_model = models.resnet50(pretrained=True, progress=True)
        self.num_classes = model_params['num_classes']
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test
        if self.num_classes != 1000:
            self.core_model.fc = nn.Linear(self.core_model.fc.in_features, self.num_classes)
            self.fc_trained = False
        else:
            self.fc_trained = True

    def core_bottlenecks(self):
        return chain(self.core_model.layer1, self.core_model.layer2, self.core_model.layer3, self.core_model.layer4)

    def forward(self, x):
        return self.core_model(x)

    def modules(self):
        # it is necessary to overwrite this as code in profiler.py depends on the order of items in this iterator
        ord_modules = []
        ord_modules.append(self.core_model.conv1)
        ord_modules.append(self.core_model.bn1)
        ord_modules.append(self.core_model.relu)
        ord_modules.append(self.core_model.maxpool)
        for i, module in enumerate(self.core_bottlenecks()):
            ord_modules.append(module)
        ord_modules.append(self.core_model.avgpool)
        ord_modules.append(self.core_model.fc)
        return chain.from_iterable(m.modules() for m in ord_modules)

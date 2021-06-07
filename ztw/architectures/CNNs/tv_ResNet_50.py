import torch
from torch import nn
from torchvision import models
from itertools import chain

import aux_funcs as af
import model_funcs as mf


class ResNet50(nn.Module):
    def __init__(self, args, model_params=None):
        super().__init__()

        self.args = args
        self.core_model = models.resnet50(pretrained=True, progress=True)
        self.num_classes = self.core_model.fc.out_features
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test
        

    def forward(self, x):
        return self.core_model(x)

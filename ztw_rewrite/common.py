import argparse
import math
from pathlib import Path

import torch
from torch import nn, optim, distributions

from architectures.condconv import CondResNet50, CondResNet18
from architectures.custom import FCNet, DCNet
from architectures.efficientnet import EfficientNetV2
from architectures.resnets import ResNet50, ResNet101, ResNet152, ResNet34, ResNet18
from architectures.vgg import VGG16BN
from architectures.wide_resnets import WideResNet
from methods.early_exits.sdn import SDN
from methods.early_exits.ztw import ZTWCascading, ZTWEnsembling


def get_lrs(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


def some_init_fun(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def wide_resnet_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=2 ** (1 / 2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def vgg_init(model):
    for m in model.modules():
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


def mixup(x, y, alpha):
    lam = distributions.beta.Beta(alpha, alpha).sample()
    indices_perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[indices_perm]
    y_a, y_b = y, y[indices_perm]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_pred, y_a, y_b, lam, criterion):
    return lam * criterion(y_pred, y_a) + (1 - lam) * criterion(y_pred, y_b)


parser = argparse.ArgumentParser()
parser.add_argument('--exp_id',
                    help='experiment id',
                    type=int,
                    default=0)
parser.add_argument('--runs_dir',
                    help='directory to save the results to',
                    type=Path,
                    default=Path.cwd() / 'runs')
parser.add_argument('--model_class',
                    help='class of the model to train',
                    type=str,
                    required=True)
parser.add_argument('--model_args',
                    help='arguments to be passed to the model init function',
                    type=str)
parser.add_argument('--dataset',
                    help='dataset to train on',
                    type=str,
                    required=True)
parser.add_argument('--init_fun',
                    help='parameters init function to be used',
                    type=str)
parser.add_argument('--batch_size',
                    help='batch size for training',
                    type=int)
parser.add_argument('--loss_type',
                    help='loss function to be used for training',
                    type=str)
parser.add_argument('--loss_args',
                    help='arguments to be passed to the loss init function',
                    type=str)
parser.add_argument('--optimizer_class',
                    help='class of the optimizer to use for training',
                    type=str)
parser.add_argument('--optimizer_args',
                    help='arguments to be passed to the optimizer init function',
                    type=str)
parser.add_argument('--scheduler_class',
                    help='class of the scheduler to use for training',
                    type=str,
                    default=None)
parser.add_argument('--scheduler_args',
                    help='arguments to be passed to the scheduler init function',
                    type=str)
parser.add_argument('--mixup_alpha',
                    help='alpha parameter for mixup\'s beta distribution',
                    type=float,
                    default=0.0)
parser.add_argument('--epochs',
                    help='number of epochs to train for',
                    type=int,
                    required=True)
parser.add_argument('--eval_points',
                    help='number of short evaluations on the validation/test data while training',
                    type=int,
                    default=100)
parser.add_argument('--eval_batches',
                    help='number of bathes to evaluate on each time while training',
                    type=int,
                    default=2)
parser.add_argument('--eval_thresholds',
                    help='number of early exit thresholds to evaluate',
                    type=int,
                    default=80)
parser.add_argument('--save_every',
                    help='save model every N minutes',
                    type=int,
                    default=10)
parser.add_argument('--use_wandb',
                    help='use weights and biases',
                    action='store_true')
# method specific args
parser.add_argument('--base_on',
                    help='unique experiment name to use the model from',
                    type=str,
                    default=None)
parser.add_argument('--with_backbone',
                    help='whether to train the backbone network along with the heads',
                    action='store_true')
parser.add_argument('--sequentially',
                    help='whether to train heads sequentially',
                    action='store_true')
parser.add_argument('--auxiliary_loss_type',
                    help='type of the auxiliary loss for early-exit heads',
                    type=str,
                    default=None)
parser.add_argument('--auxiliary_loss_weight',
                    help='weight of the auxiliary loss for early-exit heads',
                    type=float,
                    default=None)
parser.add_argument('--k_fractions',
                    help='list of fractions of channels to use for condconv',
                    type=float,
                    nargs='*',
                    default=None)

INIT_NAME_MAP = {
    'some': some_init_fun,
    'wideresnet_init': wide_resnet_init,
    'vgg_init': vgg_init,
    None: None,
}

LOSS_NAME_MAP = {
    'ce': nn.CrossEntropyLoss,
    'nll': nn.NLLLoss,
}

OPTIMIZER_NAME_MAP = {
    'sgd': optim.SGD,
    'adam': optim.AdamW,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
}

MODEL_NAME_MAP = {
    'fcnet': FCNet,
    'dcnet': DCNet,
    'vgg16bn': VGG16BN,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'condresnet18': CondResNet18,
    'condresnet50': CondResNet50,
    'wideresnet': WideResNet,
    'efficientnetv2': EfficientNetV2,
    'sdn': SDN,
    'ztw_cascading': ZTWCascading,
    'ztw_ensembling': ZTWEnsembling,
}

import math

import omegaconf
import torch.nn
from torch import nn, optim

from architectures.custom import FCNet, DCNet
from architectures.early_exits.gpf import GPF
from architectures.early_exits.l2w import L2W
from architectures.early_exits.pbee import PBEE
from architectures.early_exits.sdn import SDN
from architectures.early_exits.ztw import ZTWCascading, ZTWEnsembling
from architectures.efficientnet import EfficientNetV2
from architectures.pretrained import get_efficientnet_v2_s, get_vit_b_16, get_efficientnet_b0, get_convnext_t, \
    get_swin_v2_s
from architectures.resnets import ResNet50, ResNet101, ResNet152, ResNet34, ResNet18
from architectures.vgg import VGG16BN
from architectures.vit import VisionTransformer
from architectures.wide_resnets import WideResNet
from utils import BCEWithLogitsLossWrapper


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


def get_default_args():
    default_args = omegaconf.OmegaConf.create()

    default_args.exp_id = 0  # experiment id
    default_args.runs_dir = "runs"  # directory to save the results to
    default_args.model_class = None  # class of the model to train
    default_args.model_args = None  # arguments to be passed to the model init function
    default_args.dataset = None  # dataset to train on
    default_args.dataset_args = None  # customization arguments for the dataset
    default_args.mixup_alpha = None  # alpha parameter for mixup's beta distribution
    default_args.cutmix_alpha = None  # alpha parameter for cutmix's beta distribution
    default_args.mixup_mode = None  # how to apply mixup/cutmix ('batch', 'pair' or 'elem')
    default_args.mixup_smoothing = None  # label smoothing when using mixup
    default_args.init_fun = None  # parameters init function to be used
    default_args.batch_size = None  # batch size for training
    default_args.loss_type = None  # loss function to be used for training
    default_args.loss_args = None  # arguments to be passed to the loss init function
    default_args.optimizer_class = None  # class of the optimizer to use for training
    default_args.optimizer_args = None  # arguments to be passed to the optimizer init function
    default_args.scheduler_class = None  # class of the scheduler to use for training
    default_args.scheduler_args = None  # arguments to be passed to the scheduler init function
    default_args.clip_grad_norm = None  # gradient clipping norm
    default_args.epochs = None  # number of epochs to train for
    default_args.mixed_precision = None  # whether to use accelerate's mixed precision
    default_args.eval_points = 100  # number of short evaluations on the validation/test data while training
    default_args.eval_batches = 10  # number of batches to evaluate on each time while training
    default_args.save_every = 10  # save model every N minutes
    default_args.use_wandb = False  # use weights and biases

    # use only None for method specific args, and fill them with default values in the code!
    # unless they are not used in generate_run_name(), changing the defaults changes run names for unrelated runs!
    # method specific args
    default_args.base_on = None  # unique experiment name to use the model from

    # Early Exit specific
    default_args.with_backbone = None  # whether to train the backbone network along with the heads
    default_args.auxiliary_loss_type = None  # type of the auxiliary loss for early-exit heads
    default_args.auxiliary_loss_weight = None  # weight of the auxiliary loss for early-exit heads
    default_args.eval_thresholds = None  # number of early exit thresholds to evaluate
    # L2W
    default_args.l2w_meta_interval = None  # every n-th batch will have meta step enabled
    default_args.wpn_width = None  # weight prediction network hidden layers width
    default_args.wpn_depth = None  # weight prediction network depth
    default_args.wpn_optimizer_class = None  # class of the optimizer to use for training the weight prediction network
    default_args.wpn_optimizer_args = None  # arguments to be passed to the WPN optimizer init function
    default_args.wpn_scheduler_class = None  # class of the scheduler to use for training the weight prediction network
    default_args.wpn_scheduler_args = None  # arguments to be passed to the WPN scheduler init function
    default_args.l2w_epsilon = None  # epsilon which weights the pseudo weights (0.3 used in the paper)
    default_args.l2w_target_p = None  # target p that determines the head budget allocation (15 used in the paper)

    return default_args


INIT_NAME_MAP = {
    'wideresnet': wide_resnet_init,
    'vgg': vgg_init,
    None: None,
}

ACTIVATION_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'leaky_relu': torch.nn.LeakyReLU,
    'softplus': torch.nn.Softplus,
    'silu': torch.nn.SiLU,
    'identity': torch.nn.Identity,
}

LOSS_NAME_MAP = {
    'ce': nn.CrossEntropyLoss,
    'bcewl': BCEWithLogitsLossWrapper,
    'bce': nn.BCELoss,
    'nll': nn.NLLLoss,
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'huber': nn.HuberLoss,
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
    'wideresnet': WideResNet,
    'efficientnetv2': EfficientNetV2,
    'vit': VisionTransformer,
    'sdn': SDN,
    'pbee': PBEE,
    'gpf': GPF,
    'l2w': L2W,
    'ztw_cascading': ZTWCascading,
    'ztw_ensembling': ZTWEnsembling,
    'tv_convnext_t': get_convnext_t,
    'tv_efficientnet_b0': get_efficientnet_b0,
    'tv_efficientnet_s': get_efficientnet_v2_s,
    'tv_vit_b_16': get_vit_b_16,
    'tv_swin_v2_s': get_swin_v2_s,
}

import omegaconf
import torch.nn
from torch import nn, optim

from architectures.early_exits.gpf import GPF
from architectures.early_exits.l2w import L2W
from architectures.early_exits.pbee import PBEE
from architectures.early_exits.sdn import SDN
from architectures.early_exits.ztw import ZTWCascading, ZTWEnsembling
from architectures.pretrained import get_bert_base, get_bert_large, get_roberta_base, get_distilbert_base
from utils import BCEWithLogitsLossWrapper


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
    default_args.with_backbone = None  # whether to train the backbone network along with the heas
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
    'sdn': SDN,
    'pbee': PBEE,
    'gpf': GPF,
    'l2w': L2W,
    'ztw_cascading': ZTWCascading,
    'ztw_ensembling': ZTWEnsembling,
    'bert_base': get_bert_base,
    'bert_large': get_bert_large,
    'distilbert_base': get_distilbert_base,
    'roberta_base': get_roberta_base,
}

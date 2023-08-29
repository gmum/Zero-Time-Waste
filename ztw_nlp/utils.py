import base64
import hashlib
import logging
import numbers
import os
import warnings
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Union, Tuple, List, Dict, Set, FrozenSet, Callable, Iterator, Any, Optional

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import patch_environment, PrepareForLaunch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import elementwise_flop_counter, get_shape
from omegaconf import OmegaConf
from torch import nn, Tensor, inf


def clip_by_norm_(tensors, max_norm: float, norm_type: float = 2.0):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.)
    if norm_type == inf:
        norms = [t.detach().abs().max() for t in tensors]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = [torch.norm(t, norm_type) for t in tensors]
        total_norm = torch.norm(torch.stack([norm for norm in norms]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for t in tensors:
        t.detach().mul_(clip_coef_clamped)
    return total_norm


class BCEWithLogitsLossWrapper(nn.BCEWithLogitsLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.dim() - 1 == target.dim():
            target = nn.functional.one_hot(target, num_classes=input.size(-1)).float()
        return nn.functional.binary_cross_entropy_with_logits(input, target,
                                                              self.weight,
                                                              pos_weight=self.pos_weight,
                                                              reduction=self.reduction)


# mixup code from:
# https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/data/mixup.py#L90


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        if len(x) % 2 != 0:
            warnings.warn('Batch size is not even as required by mixup - skipping last element.')
            x, target = x[:-1], target[:-1]
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target


def get_lrs(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


def unfrozen_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad == True)


def find_module_names(model: nn.Module, filter: Callable[[nn.Module, nn.Module], bool]):
    found_names = []
    for name, module in model.named_modules():
        if filter(model, module):
            found_names.append(name)
    return found_names


def get_module_name(module: nn.Module, submodule: nn.Module):
    for name, m in module.named_modules():
        if m is submodule:
            return name


def get_module_by_name(module: nn.Module, name: str):
    if not name:
        return module
    else:
        names = name.split(sep='.')
        return reduce(getattr, names, module)


def get_parent_module_name(name: str):
    names = name.split(sep='.')
    return '.'.join(names[:-1])


def set_module_by_name(module: nn.Module, name: str, replacement):
    names = name.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], replacement)


def set_parameter_by_name(module: nn.Module, name: str, replacement):
    names = name.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    parent._parameters[names[-1]] = replacement


def add_save_activations_hook(model, module_names):
    module_inputs = {}
    module_outputs = {}
    module_id_to_name = {}
    hook_handles = []
    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_activations_hook(m, input, output):
            module_name = module_id_to_name[id(m)]
            # TODO will multiple inputs/outputs ever be used?
            module_inputs[module_name] = input
            module_outputs[module_name] = output

        handle = module.register_forward_hook(save_activations_hook)
        hook_handles.append(handle)
    return module_inputs, module_outputs, hook_handles


def add_save_outputs_hook(model, module_names):
    gating_outputs = {}
    module_id_to_name = {}
    hook_handles = []

    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_gating_hook(m, _input, output):
            module_name = module_id_to_name[id(m)]
            # TODO will multiple inputs/outputs ever be used?
            gating_outputs[module_name] = output

        handle = module.register_forward_hook(save_gating_hook)
        hook_handles.append(handle)
    return gating_outputs, hook_handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def count_scaled_dot_product_attention_ops(inputs: List[Any], _outputs: List[Any]) -> Counter[str]:
    q_samples, q_heads, q_tokens, q_dim = get_shape(inputs[0])
    k_samples, k_heads, k_tokens, k_dim = get_shape(inputs[1])
    v_samples, v_heads, v_tokens, v_dim = get_shape(inputs[2])
    assert q_dim == k_dim
    assert q_tokens == k_tokens == v_tokens
    assert q_heads == k_heads == v_heads
    assert q_samples == k_samples == v_samples
    # query @ key.transpose(-2, -1)
    q_k_matmul_ops = q_samples * q_heads * q_tokens * q_tokens * q_dim
    # / math.sqrt(query.size(-1))
    div_ops = q_samples * q_heads * q_tokens * q_tokens
    # + attn_mask
    if get_shape(inputs[3]) is not None or isinstance(inputs[3], numbers.Number):
        mask_add_ops = q_samples * q_heads * q_tokens * q_tokens
    else:
        mask_add_ops = 0
    # torch.softmax(..., dim=-1)
    softmax_ops = 2 * q_samples * q_heads * q_tokens * q_tokens
    # attn_weight @ value
    v_matmul_ops = q_samples * q_heads * q_tokens * q_tokens * v_dim
    return Counter({
        'matmul': q_k_matmul_ops + v_matmul_ops,
        'div': div_ops,
        'add': mask_add_ops,
        'softmax': softmax_ops
    })


OP_HANDLERS = {
    'aten::add': elementwise_flop_counter(0, 1),
    'aten::add_': elementwise_flop_counter(0, 1),
    'aten::radd': elementwise_flop_counter(0, 1),
    'aten::sub': elementwise_flop_counter(0, 1),
    'aten::sub_': elementwise_flop_counter(0, 1),
    'aten::rsub': elementwise_flop_counter(0, 1),
    'aten::mul': elementwise_flop_counter(0, 1),
    'aten::mul_': elementwise_flop_counter(0, 1),
    'aten::rmul': elementwise_flop_counter(0, 1),
    'aten::div': elementwise_flop_counter(0, 1),
    'aten::div_': elementwise_flop_counter(0, 1),
    'aten::rdiv': elementwise_flop_counter(0, 1),
    'aten::exp': elementwise_flop_counter(0, 1),
    'aten::cumsum': elementwise_flop_counter(0, 1),
    'aten::ne': elementwise_flop_counter(0, 1),
    'aten::gelu': elementwise_flop_counter(0, 1),
    'aten::silu_': elementwise_flop_counter(0, 1),
    'aten::dropout_': elementwise_flop_counter(0, 1),
    'aten::sigmoid': elementwise_flop_counter(0, 1),
    'aten::softmax': elementwise_flop_counter(0, 2),
    'aten::log_softmax': elementwise_flop_counter(0, 2),
    'aten::argmax': elementwise_flop_counter(0, 1),
    'aten::one_hot': elementwise_flop_counter(0, 1),
    'aten::flatten': elementwise_flop_counter(0, 0),
    'aten::unflatten': elementwise_flop_counter(0, 0),
    'aten::mean': elementwise_flop_counter(1, 0),
    'aten::sum': elementwise_flop_counter(1, 0),
    'aten::topk': elementwise_flop_counter(1, 1),
    'aten::scatter': elementwise_flop_counter(1, 1),
    'aten::gather': elementwise_flop_counter(1, 1),
    'aten::adaptive_max_pool2d': elementwise_flop_counter(1, 0),
    # custom
    'aten::scaled_dot_product_attention': count_scaled_dot_product_attention_ops,
    # 'aten::_native_multi_head_attention': count_native_attention_ops,
}


def flop_count(model: torch.nn.Module, input) -> FlopCountAnalysis:
    return FlopCountAnalysis(model, input).set_op_handle(**OP_HANDLERS)


def get_loader(data: torch.utils.data.Dataset, batch_size: int, accelerator: Accelerator,
               shuffle: bool = True, num_workers: int = 8, pin: bool = True):
    loader = torch.utils.data.DataLoader(dataset=data,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         pin_memory=pin,
                                         num_workers=num_workers)
    return accelerator.prepare(loader)


def make_hashable(o: Union[Tuple, List, Dict, Set, FrozenSet]) -> Union[Tuple, List, Dict, Set, FrozenSet]:
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))
    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))
    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))
    return o


def make_hash_sha256(o: Union[Tuple, List, Dict, Set, FrozenSet]) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b32encode(hasher.digest()).decode()


def generate_run_name(args) -> Tuple[str, str]:
    # Properties from the config that are NOT to be included into hashing.
    omitted_keys = ['exp_id', 'runs_dir', 'eval_points', 'eval_batches', 'save_every', 'use_wandb', 'eval_thresholds']
    args_dict = OmegaConf.to_container(args, resolve=True)
    hashed_flags = {k: v for k, v in args_dict.items() if k not in omitted_keys and v is not None}
    short_hash = make_hash_sha256(hashed_flags)[:8]
    exp_name = f'{args.dataset}_{args.model_class}_{short_hash}'
    run_name = f'{exp_name}_{args.exp_id}'
    logging.info(f'Generated run name: {run_name}')
    return exp_name, run_name


def get_run_id(run_name: str):
    api = wandb.Api()
    entity = os.environ['WANDB_ENTITY']
    project = os.environ['WANDB_PROJECT']
    retrieved_runs = api.runs(f'{entity}/{project}', filters={'display_name': run_name})
    logging.info(f'Retrieved {len(retrieved_runs)} for run_name: {run_name}')
    assert len(retrieved_runs) <= 1, f'retrieved_runs: {retrieved_runs}'
    if len(retrieved_runs) == 1:
        return retrieved_runs[0].id


def load_state(accelerator: Accelerator, state_path: Path):
    # TODO add wandb support?
    if state_path.exists() and state_path.is_dir():
        accelerator.load_state(state_path)
    elif accelerator.is_main_process:
        logging.info(f'No state file found - training from scratch.')


def save_state(accelerator: Accelerator, state_path: Path):
    # TODO add wandb support?
    accelerator.save_state(state_path)


def retrieve_final(args, run_name: str, device: Union[torch.device, str] = 'cpu'):
    final_path = args.runs_dir / run_name / f'final.pth'
    if final_path.exists() and final_path.is_file():
        state = torch.load(final_path, map_location=device)
        logging.info(f'Loaded state for {run_name} from {str(final_path)}')
    elif args.use_wandb:
        api = wandb.Api()
        entity = os.environ['WANDB_ENTITY']
        project = os.environ['WANDB_PROJECT']
        retrieved_runs = api.runs(f'{entity}/{project}', filters={'display_name': run_name})
        assert len(retrieved_runs) <= 1, f'retrieved_runs: {retrieved_runs}'
        assert len(retrieved_runs) > 0, f'Run {run_name} not found'
        run = retrieved_runs[0]
        run.file('final.pth').download(root=str(final_path.parent), replace=True)
        state = torch.load(final_path, map_location=device)
        logging.info(f'Loaded state for {run_name} from W&B')
    else:
        raise FileNotFoundError('Cannot find the final.pth file')
    return state


def save_final(args, final_path, final_results):
    torch.save(final_results, final_path)
    logging.info(f'Saved final results to {str(final_path)}')
    # if args.use_wandb:
    # wandb.save(str(final_path))
    # logging.info(f'Saved final results to W&B')


def create_model(model_class, model_args):
    from common import MODEL_NAME_MAP
    return MODEL_NAME_MAP[model_class](**model_args)


def load_model(args, exp_name: str, exp_id: str, device: Union[torch.device, str] = 'cpu'):
    run_states = []
    run_args = []
    run_model_args = []
    while exp_name is not None:
        state = retrieve_final(args, f'{exp_name}_{exp_id}', device)
        arg = state['args']
        model_arg = arg.model_args
        run_states.append(state)
        run_args.append(arg)
        run_model_args.append(model_arg)
        exp_name = arg.base_on if hasattr(arg, 'base_on') else None
        exp_id = arg.exp_id
    # create model from the most nested base model
    for arg, model_arg in zip(reversed(run_args), reversed(run_model_args)):
        if arg.base_on is None:
            model = create_model(arg.model_class, model_arg).to(device)
        else:
            model_arg = {'base_model': model, **model_arg}
            model = create_model(arg.model_class, model_arg).to(device)
    # state loading is only necessary for the model being loaded
    state_dict = run_states[0]['model_state']
    model.load_state_dict(state_dict)
    return model, run_args[0], run_states[0]


def load_run(args, run_name: str, device: Union[torch.device, str] = 'cpu'):
    split_run_name = run_name.split('_')
    exp_id = split_run_name[-1]
    exp_name = '_'.join(split_run_name[:-1])
    return load_model(args, exp_name, exp_id, device)


def accelerate_launcher(function, num_processes=None, mixed_precision="no", use_port="29500"):
    from torch.multiprocessing import start_processes
    from torch.multiprocessing.spawn import ProcessRaisedException
    if len(AcceleratorState._shared_state) > 0:
        raise ValueError(
            "To launch a multi-GPU training, the `Accelerator` should only be initialized "
            "inside your training function."
        )
    if torch.cuda.is_initialized():
        raise ValueError(
            "To launch a multi-GPU training, you need to avoid running any instruction "
            "using `torch.cuda` before starting DDP/accelerate."
        )
    with patch_environment(
            world_size=num_processes, master_addr="127.0.0.1", master_port=use_port, mixed_precision=mixed_precision
    ):
        launcher = PrepareForLaunch(function, distributed_type="MULTI_GPU")
        print(f"Launching training on {num_processes} GPUs.")
        try:
            start_processes(launcher, nprocs=num_processes, start_method="spawn")
            print(f"All processes exited")
        except ProcessRaisedException as e:
            print(f"ProcessRaisedException: {e}")


def submit_job(executor, job_func, *job_args, num_gpus=1):
    return executor.submit(job_func, *job_args)

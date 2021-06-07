import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from ztw import InternalClassifier, RunningEnsemble

# adapted from: https://github.com/yigitcankaya/Shallow-Deep-Networks/blob/master/profiler.py


def count_conv2d(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops * m.groups

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_softmax(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_maxpool(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_adaptive_maxpool(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    # https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/aten/src/ATen/native/AdaptiveMaxPooling2d.cpp
    x = x[0]
    est_kernel_size = math.ceil(x.size(2) / y.size(2))
    kernel_ops = torch.prod(torch.Tensor([est_kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_adaptive_avgpool(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    # https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/aten/src/ATen/native/AdaptiveAveragePooling.cpp
    x = x[0]
    est_kernel_size = math.ceil(x.size(2) / y.size(2))
    total_add = torch.prod(torch.Tensor([est_kernel_size])) - 1

    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_re(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    x = x[0]
    # x shape is [batch_size=1, head_index, input_dim]
    batch_size = x.size(0)
    num_heads_up_to = x.size(-2)
    num_classes = x.size(-1)
    total_ops = 0
    # log softmax ops
    # - b
    total_ops += x.numel()
    # exp
    total_ops += x.numel()
    # sum
    total_ops += (num_classes - 1) * num_heads_up_to * batch_size
    # div 
    total_ops += x.numel()
    # exp weights
    total_ops += m.weight.numel()
    # x * resized_weight ops
    total_ops += x.numel()
    # x.mean(1) ops
    # sum
    total_ops += (num_classes - 1) * num_heads_up_to * batch_size
    # div
    total_ops += x.numel()
    # + self.bias ops
    total_ops += num_classes * batch_size
    m.total_ops += torch.Tensor([int(total_ops)])


def profile_ee(model: nn.Module,
               input_size: Tuple[int],
               device: Union[torch.device, str] = 'cpu') -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    inp = (1, *input_size)
    re_present = False

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
            m.register_forward_hook(count_adaptive_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            m.register_forward_hook(count_adaptive_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        elif isinstance(m, RunningEnsemble):
            m.register_forward_hook(count_re)
            re_present = True
        else:
            print('Profiling not implemented for ', m)

    model.to(device)
    for m in model.modules():
        add_hooks(m)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    output_total_ops = {}
    output_total_params = {}

    total_ops = 0
    total_params = 0

    current_head = 0
    internal_c_id = -1
    internal_c_modules_left = 0

    def mark_ic_descendant(descendant):
        nonlocal internal_c_modules_left
        nonlocal internal_c_id
        if len(list(descendant.children())) > 0: return
        internal_c_modules_left += 1
        descendant.internal_c_id = internal_c_id

    for layer_id, m in enumerate(model.modules()):
        if isinstance(m, (InternalClassifier)):
            assert internal_c_id == -1
            internal_c_id = id(m)
            m.apply(mark_ic_descendant)

        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params

        if internal_c_id != -1 and internal_c_modules_left > 0:
            assert m.internal_c_id == internal_c_id
            internal_c_modules_left -= 1
            del m.internal_c_id
            # if the entire head has been classified
            # then add current ops/params to outputs dict
            if internal_c_modules_left == 0 and not re_present:
                internal_c_id = -1
                output_total_ops[current_head] = total_ops.numpy()[0] / 1e9
                output_total_params[current_head] = total_params.numpy()[0] / 1e6
                current_head += 1

        if re_present and isinstance(m, RunningEnsemble):
            output_total_ops[current_head] = total_ops.numpy()[0] / 1e9
            output_total_params[current_head] = total_params.numpy()[0] / 1e6
            current_head += 1

    output_total_ops[current_head] = total_ops.numpy()[0] / 1e9
    output_total_params[current_head] = total_params.numpy()[0] / 1e6

    return output_total_ops, output_total_params

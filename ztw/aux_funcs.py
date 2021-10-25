# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import copy
import itertools as it
import matplotlib
import numpy as np
import os
import os.path
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

import re
from bisect import bisect_right

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler

import network_architectures as arcs
from data import CIFAR10, CIFAR100, ImageNet, TinyImagenet, OCT2017
from profiler import profile, profile_sdn
from architectures.SDNs.tv_ResNet_50_SDN import ResNet50_SDN


# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    # sys.stderr = Logger(log_file, 'err')


# the learning rate scheduler
class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of' ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr, 8)
            lrs.append(new_lr)
        return lrs


# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size / 4)
    else:
        return -1


class SDNPool(nn.Module):
    def __init__(self, input_size, input_channels, pool_size=None):
        super().__init__()
        # get the pooling size
        red_kernel_size = feature_reduction_formula(input_size)
        if pool_size is None and red_kernel_size == -1:
            self.forward = self.forward_wo_pooling
            self.after_pool_dim = input_channels * input_size * input_size
        else:
            if pool_size is None:
                red_input_size = int(input_size / red_kernel_size)
            else:
                red_input_size = pool_size
                red_kernel_size = int(input_size / pool_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.forward = self.forward_w_pooling
            self.after_pool_dim = input_channels * red_input_size * red_input_size

    def forward_w_pooling(self, x):
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        mixed = mixed.view(mixed.size(0), -1)
        return mixed

    def forward_wo_pooling(self, x):
        return x


class SDNPool_v2(nn.Module):
    def __init__(self, input_size, input_channels, pool_size=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1))
        if pool_size is None:
            pool_size = 4
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.after_pool_dim = input_channels * pool_size * pool_size

    def forward(self, x):
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        mixed = mixed.view(mixed.size(0), -1)
        return mixed


# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self,
                 args,
                 input_size,
                 output_channels,
                 num_classes,
                 alpha=0.5,
                 head_variant=None,
                 pool_size=4,
                 heads_per_ensemble=1,
                 prev_dim=0):
        assert head_variant is not None
        super(InternalClassifier, self).__init__()

        # self.output_channels = output_channels
        # self.head_variant = head_variant

        self.conv_layers_ens = [list() for _ in range(heads_per_ensemble)]
        self.fc_layers_ens = [list() for _ in range(heads_per_ensemble)]
        self.detach = args.detach_prev
        if args.detach_norm == "layernorm" and prev_dim > 0:
            self.detach_norm = nn.LayerNorm(prev_dim)
        else:
            self.detach_norm = nn.Identity()

        for idx in range(heads_per_ensemble):
            input_dim = None
            for layer_type in head_variant:
                # some hacks ahead
                if layer_type == 'conv' or layer_type == 'conv_less_ch':
                    input_channels = output_channels
                    if layer_type == 'conv_less_ch':
                        output_channels = input_channels // 4
                    if args.dataset == 'imagenet' or args.dataset == 'oct2017':
                        if input_size > 8:
                            stride, padding = 2, 1
                            input_size = input_size // 2
                        else:
                            stride, padding = 1, 1
                            output_channels = input_channels // 4
                        self.conv_layers_ens[idx].append(
                            nn.Conv2d(input_channels,
                                      output_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=padding,
                                      bias=True))
                    elif args.dataset == 'tinyimagenet':
                        padding = 2 if input_size > 8 else 1
                        self.conv_layers_ens[idx].append(
                            nn.Conv2d(output_channels,
                                      output_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=padding,
                                      bias=True))
                        input_size = input_size // 2
                    else:
                        self.conv_layers_ens[idx].append(
                            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
                    self.conv_layers_ens[idx].append(nn.ReLU())
                elif layer_type == 'max_pool':
                    self.conv_layers_ens[idx].append(nn.AdaptiveMaxPool2d((args.size_after_pool, args.size_after_pool)))
                    self.conv_layers_ens[idx].append(nn.Flatten())
                    input_dim = output_channels * args.size_after_pool * args.size_after_pool + prev_dim
                elif layer_type == 'avg_pool':
                    self.conv_layers_ens[idx].append(nn.AdaptiveAvgPool2d((args.size_after_pool, args.size_after_pool)))
                    self.conv_layers_ens[idx].append(nn.Flatten())
                    input_dim = output_channels * args.size_after_pool * args.size_after_pool + prev_dim
                elif layer_type == 'sdn_pool':
                    if args.arch == 'tv_resnet':
                        sdn_pool = SDNPool_v2(input_size, output_channels, pool_size=args.size_after_pool)
                    else:
                        sdn_pool = SDNPool(input_size, output_channels, pool_size=args.size_after_pool)
                    self.conv_layers_ens[idx].append(sdn_pool)
                    self.conv_layers_ens[idx].append(nn.Flatten())
                    # red_kernel_size = -1 # to test the effects of the feature reduction
                    input_dim = sdn_pool.after_pool_dim + prev_dim
                elif layer_type == 'linear':
                    assert input_dim is not None
                    self.fc_layers_ens[idx].append(nn.Linear(input_dim, input_dim))
                    self.fc_layers_ens[idx].append(nn.ReLU())
            # always add the last fc layer
            assert input_dim is not None, f'head_variant: {head_variant} args.head_arch: {args.head_arch}'
            self.fc_layers_ens[idx].append(nn.Linear(input_dim, num_classes))

            self.conv_layers_ens[idx] = nn.ModuleList(self.conv_layers_ens[idx])
            self.fc_layers_ens[idx] = nn.ModuleList(self.fc_layers_ens[idx])
        self.conv_layers_ens = nn.ModuleList(self.conv_layers_ens)
        self.fc_layers_ens = nn.ModuleList(self.fc_layers_ens)

    def forward(self, in_x, prev_output):
        if prev_output is not None:
            if self.detach:
                prev_output = [o.detach() for o in prev_output]
            prev_output = [self.detach_norm(o) for o in prev_output]
        outputs = []
        for idx, (conv_layers, fc_layers) in enumerate(zip(self.conv_layers_ens, self.fc_layers_ens)):
            x = in_x
            for conv_layer in conv_layers:
                x = conv_layer(x)
            if prev_output is not None:
                x = torch.cat([x, prev_output[idx]], -1)
            for fc_layer in fc_layers:
                x = fc_layer(x)
            outputs += [x]
        return outputs


def get_random_seed():
    return 1221  # 121 and 1221


def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))


def set_rng_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])


def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label,
                           title):
    plt.hist([hist_first_values, hist_second_values], bins=25, label=[first_label, second_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()


def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)

    for output in outputs:
        cur_disagreement = nn.functional.pairwise_distance(outputs[-1], output, p=p)
        cur_disagreement = cur_disagreement.to(device)
        for instance_id in range(outputs[0].size(0)):
            confusion_scores[instance_id] += cur_disagreement[instance_id]

    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0]  # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1]  # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores


def get_dataset(args, dataset, batch_size=128, add_trigger=False):
    if dataset == 'cifar10':
        return load_cifar10(args, batch_size, add_trigger)
    elif dataset == 'cifar100':
        return load_cifar100(args, batch_size)
    elif dataset == 'tinyimagenet':
        return load_tinyimagenet(args, batch_size)
    elif dataset == 'imagenet':
        return ImageNet(batch_size // 2)
    elif dataset == 'oct2017':
        return OCT2017(batch_size // 2)


def load_cifar10(args, batch_size, add_trigger=False):
    cifar10_data = CIFAR10(batch_size=batch_size,
                           add_trigger=add_trigger,
                           examples_num=args.examples_num,
                           validation=args.validation_dataset)
    return cifar10_data


def load_cifar100(args, batch_size):
    cifar100_data = CIFAR100(batch_size=batch_size, examples_num=args.examples_num, validation=args.validation_dataset)
    return cifar100_data


def load_tinyimagenet(args, batch_size):
    tiny_imagenet = TinyImagenet(batch_size=batch_size,
                                 examples_num=args.examples_num,
                                 validation=args.validation_dataset)
    return tiny_imagenet


def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    # output_depths.append(total_depth)

    return np.array(output_depths) / total_depth, total_depth


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def model_exists(models_path, model_name):
    return os.path.isdir(models_path + '/' + model_name)


def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]


def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']


def get_full_optimizer(model, lr_params, stepsize_params):
    lr = lr_params[0]
    weight_decay = lr_params[1]
    momentum = lr_params[2]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def get_sdn_ic_only_optimizer(model, lr_params, stepsize_params):
    lr = lr_params[0]
    weight_decay = lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            model_params = arcs.load_params(models_path, model_name, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            architecture = model_params['architecture']
            print(model_name)
            task = model_params['task']
            print(task)
            net_type = model_params['network_type']
            print(net_type)

            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            top5_test = model_params['test_top5_acc']
            top5_train = model_params['train_top5_acc']

            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = arcs.load_model(args, models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                if architecture == 'dsn':
                    total_ops, total_params = profile_sdn(model, input_size, device)
                    print("#Ops (GOps): {}".format(total_ops))
                    print("#Params (mil): {}".format(total_params))

                else:
                    total_ops, total_params = profile(model, input_size, device)
                    print("#Ops: %f GOps" % (total_ops / 1e9))
                    print("#Parameters: %f M" % (total_params / 1e6))

            print('------------------------')
        except:
            print('FAIL: {}'.format(model_name))
            continue


def sdn_prune(sdn_path, sdn_name, prune_after_output, epoch=-1, preloaded=None):
    print('Pruning an SDN...')

    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(args, sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    output_layer = get_nth_occurance_index(sdn_model.add_output, prune_after_output)

    pruned_model = copy.deepcopy(sdn_model)
    pruned_params = copy.deepcopy(sdn_params)

    new_layers = nn.ModuleList()
    prune_add_output = []

    for layer_id, layer in enumerate(sdn_model.layers):
        if layer_id == output_layer:
            break
        new_layers.append(layer)
        prune_add_output.append(sdn_model.add_output[layer_id])

    last_conv_layer = sdn_model.layers[output_layer]
    end_layer = copy.deepcopy(last_conv_layer.output)

    last_conv_layer.output = nn.Sequential()
    last_conv_layer.forward = last_conv_layer.only_forward
    last_conv_layer.no_output = True
    new_layers.append(last_conv_layer)

    pruned_model.layers = new_layers
    pruned_model.end_layers = end_layer

    pruned_model.add_output = prune_add_output
    pruned_model.num_output = prune_after_output + 1

    pruned_params['pruned_after'] = prune_after_output
    pruned_params['pruned_from'] = sdn_name

    return pruned_model, pruned_params


# convert a cnn to a sdn by adding output layers to internal layers
def cnn_to_sdn(args, cnn_path, cnn_name, sdn_params, epoch=-1, preloaded=None):
    print('Converting a CNN to a SDN...')
    if preloaded is None:
        cnn_model, _ = arcs.load_model(args, cnn_path, cnn_name, epoch=epoch)
    else:
        cnn_model = preloaded

    sdn_params['architecture'] = 'sdn'
    sdn_params['converted_from'] = cnn_name
    sdn_model = arcs.get_sdn(cnn_model)(args, sdn_params)

    if hasattr(cnn_model, 'init_conv'):
        sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    if isinstance(sdn_model, ResNet50_SDN):
        sdn_model.core_model = cnn_model.core_model
    else:
        for layer_id, cnn_layer in enumerate(cnn_model.layers):
            sdn_layer = sdn_model.layers[layer_id]
            sdn_layer.layers = cnn_layer.layers
            layers.append(sdn_layer)
        sdn_model.layers = layers
        sdn_model.end_layers = cnn_model.end_layers

    return sdn_model, sdn_params


def sdn_to_cnn(sdn_path, sdn_name, epoch=-1, preloaded=None):
    print('Converting a SDN to a CNN...')
    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(args, sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    cnn_params = copy.deepcopy(sdn_params)
    cnn_params['architecture'] = 'cnn'
    cnn_params['converted_from'] = sdn_name
    cnn_model = arcs.get_cnn(sdn_model)(cnn_params)

    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)

    cnn_model.layers = layers

    cnn_model.end_layers = sdn_model.end_layers

    return cnn_model, cnn_params


ENS_PARAM_NAME_REGEX = re.compile(r'ens\.(\d+)')


def freeze_bn(module, freeze=True):
    if isinstance(module, nn.BatchNorm2d):
        if freeze:

            def train_substitute(mode: bool = True):
                super(type(module)).__thisclass__.train(module, False)
                return module

            module.train = train_substitute
        else:
            if 'train' in module.__dict__:
                del module.train


def freeze(model, mode, boosting=False):
    if mode == 'except_core':
        for name, param in model.named_parameters():
            if 'output' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for name, module in model.named_modules():
            freeze_bn(module, 'output' in name)
    elif mode == 'except_outputs':
        for name, param in model.named_parameters():
            if 'output' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, module in model.named_modules():
            freeze_bn(module, 'output' not in name)
    elif isinstance(mode, int):
        if boosting:
            current_head = -1
            currently_in_output = False
            for name, param in model.named_parameters():
                if not currently_in_output and 'output' in name:
                    current_head += 1
                    currently_in_output = True
                if currently_in_output and 'output' not in name:
                    currently_in_output = False
                if current_head == mode and currently_in_output:
                    param.requires_grad = True
                    # print(f'Unfreezing: {name}')
                else:
                    param.requires_grad = False
                    # print(f'Freezing: {name}')
            # TODO deduplicate?
            current_head = -1
            currently_in_output = False
            for name, module in model.named_modules():
                if not currently_in_output and 'output' in name:
                    current_head += 1
                    currently_in_output = True
                if currently_in_output and 'output' not in name:
                    currently_in_output = False
                if current_head == mode and currently_in_output:
                    # print(f'Unfreezing BN: {name}')
                    freeze_bn(module, False)
                else:
                    # print(f'Freezing BN: {name}')
                    freeze_bn(module)
        else:
            # (only net_i in all heads are unfrozen)
            for name, param in model.named_parameters():
                if 'output' in name:
                    matches = ENS_PARAM_NAME_REGEX.search(name)
                    if matches is not None and int(matches.group(1)) == mode:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
            # TODO deduplicate?
            for name, module in model.named_modules():
                if 'output' in name:
                    matches = ENS_PARAM_NAME_REGEX.search(name)
                    if matches is not None and int(matches.group(1)) == mode:
                        freeze_bn(module, False)
                    else:
                        freeze_bn(module)
                else:
                    freeze_bn(module)
    elif mode == 'nothing':
        for param in model.modules():
            param.requires_grad = True
        for name, module in model.named_modules():
            freeze_bn(module, False)
    elif mode == 'final_layer_only':
        for param in model.parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            freeze_bn(module)
        ord_modules = list(model.modules())
        for param in ord_modules[-1].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f'mode argument value incorrect: {mode}')

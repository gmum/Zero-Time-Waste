# network_architectures.py
# contains the functions to create and save CNNs and SDNs
# VGG, ResNet, Wide ResNet and MobileNet
# also contains the hyper-parameters for model training

import os
import os.path
import pickle

import neptune
import torch

from architectures.CNNs.MobileNet import MobileNet
from architectures.CNNs.ResNet import ResNet
from architectures.CNNs.VGG import VGG
from architectures.CNNs.WideResNet import WideResNet
from architectures.CNNs.tv_ResNet_50 import ResNet50
from architectures.SDNs.MobileNet_SDN import MobileNet_SDN
from architectures.SDNs.ResNet_SDN import ResNet_SDN
from architectures.SDNs.VGG_SDN import VGG_SDN
from architectures.SDNs.WideResNet_SDN import WideResNet_SDN
from architectures.SDNs.tv_ResNet_50_SDN import ResNet50_SDN
from architectures.weighted_avg_model import WeightedAverage


def canonical_name(args, model_name):
    postfix = f"_{'_'.join(args.head_arch)}" if args.head_arch is not None else ''
    postfix += f'_heads_{args.heads}'
    postfix += f'_seq' if args.sequential_training else ''
    postfix += f'_stacking' if args.stacking else ''
    postfix += f'_detach' if args.detach_prev else ''
    postfix += f'_e_{args.heads_per_ensemble}' if args.heads_per_ensemble > 1 else ''
    postfix += f'_boost_{args.boosting}' if args.boosting != 'off' else ''
    postfix += f'_{args.conf_reduction}' if 'confidence' in args.boosting else ''
    postfix += f'_beta_{args.beta}' if args.beta else ''
    postfix += f'_{args.suffix}' if args.suffix else ''
    return f'{model_name}_{args.loss}{postfix}'


def args_model_params(args, model_params):
    model_params['head_variant'] = args.head_arch
    model_params['heads_per_ensemble'] = args.heads_per_ensemble


def save_networks(args, model_name, model_params, models_path, save_type):
    cnn_name = model_name + '_cnn'
    sdn_name = canonical_name(args, model_name)

    if 'c' in save_type:
        print('Saving CNN...')
        model_params['architecture'] = 'cnn'
        model_params['base_model'] = cnn_name
        network_type = model_params['network_type']

        if 'wideresnet' in network_type:
            model = WideResNet(args, model_params)
        elif 'tv_resnet' in network_type:
            model = ResNet50(args, model_params)
        elif 'resnet' in network_type:
            model = ResNet(args, model_params)
        elif 'vgg' in network_type:
            model = VGG(args, model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(args, model_params)

        save_model(args, model, model_params, models_path, cnn_name, epoch=0)

    if 'd' in save_type:
        print('Saving SDN...')
        model_params['architecture'] = 'sdn'
        model_params['base_model'] = sdn_name
        network_type = model_params['network_type']

        if 'wideresnet' in network_type:
            model = WideResNet_SDN(args, model_params)
        elif 'tv_resnet' in network_type:
            model = ResNet50_SDN(args, model_params)
        elif 'resnet' in network_type:
            model = ResNet_SDN(args, model_params)
        elif 'vgg' in network_type:
            model = VGG_SDN(args, model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(args, model_params)

        save_model(args, model, model_params, models_path, sdn_name, epoch=0)

    return cnn_name, sdn_name


def create_vgg16bn(args, models_path, task, save_type, get_params=False):
    print('Creating VGG16BN untrained {} models...'.format(task))

    model_params = get_task_params(task)
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]

    model_params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_name = '{}_vgg16bn'.format(task)

    # architecture params
    model_params['network_type'] = 'vgg16bn'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True
    if args.heads == 'original':
        model_params['add_ic'] = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    elif args.heads == 'all':
        model_params['add_ic'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.heads == 'half':
        model_params['add_ic'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    args_model_params(args, model_params)

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(args, model_name, model_params, models_path, save_type)


def create_resnet56(args, models_path, task, save_type, get_params=False):
    print('Creating resnet56 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9, 9, 9]
    if args.heads == 'original':
        model_params['add_ic'] = [[0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0, 1, 0, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    elif args.heads == 'all':
        model_params['add_ic'] = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    elif args.heads == 'half':
        model_params['add_ic'] = [[1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 0, 1, 0, 1, 0, 1]]
    elif args.heads == 'full+half':
        model_params['add_ic'] = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 0, 1, 0, 1, 0, 1]]
    model_name = '{}_resnet56'.format(task)

    model_params['network_type'] = 'resnet56'
    model_params['augment_training'] = True
    model_params['init_weights'] = True
    args_model_params(args, model_params)

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(args, model_name, model_params, models_path, save_type)


def create_wideresnet32_4(args, models_path, task, save_type, get_params=False):
    print('Creating wrn32_4 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['num_blocks'] = [5, 5, 5]
    model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3

    model_name = '{}_wideresnet32_4'.format(task)

    if args.heads == 'original':
        model_params['add_ic'] = [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0,
                                                                     0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    elif args.heads == 'all':
        model_params['add_ic'] = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    elif args.heads == 'half':
        model_params['add_ic'] = [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]
    model_params['network_type'] = 'wideresnet32_4'
    model_params['augment_training'] = True
    model_params['init_weights'] = True
    args_model_params(args, model_params)

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(args, model_name, model_params, models_path, save_type)


def create_mobilenet(args, models_path, task, save_type, get_params=False):
    print('Creating MobileNet untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_mobilenet'.format(task)

    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True
    if args.heads == 'original':
        model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    elif args.heads == 'all':
        model_params['add_ic'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.heads == 'half':
        model_params['add_ic'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    args_model_params(args, model_params)

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(args, model_name, model_params, models_path, save_type)


def create_tv_resnet(args, models_path, task, save_type, get_params=False):
    print('Creating torchvision resnet pretrained {} models...'.format(task))

    model_params = get_task_params(task)
    model_name = '{}_tv_resnet'.format(task)

    model_params['network_type'] = 'tv_resnet'
    model_params['augment_training'] = True
    model_params['init_weights'] = False
    args_model_params(args, model_params)

    get_lr_params(model_params, args)

    if get_params:
        return model_params

    return save_networks(args, model_name, model_params, models_path, save_type)


def get_task_params(task: str):
    if task.startswith('cifar100'):
        return cifar100_params()
    elif task.startswith('cifar10'):
        return cifar10_params()
    elif task.startswith('tinyimagenet'):
        return tiny_imagenet_params()
    elif task.startswith('imagenet'):
        return imagenet_params()
    elif task.startswith('oct2017'):
        return oct2017_params()


def cifar10_params():
    model_params = {}
    model_params['task'] = 'cifar10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    return model_params


def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    return model_params


def tiny_imagenet_params():
    model_params = {}
    model_params['task'] = 'tinyimagenet'
    model_params['input_size'] = 64
    model_params['num_classes'] = 200
    return model_params


def imagenet_params():
    model_params = {}
    model_params['task'] = 'imagenet'
    model_params['input_size'] = 224
    model_params['num_classes'] = 1000
    return model_params


def oct2017_params():
    model_params = {}
    model_params['task'] = 'oct2017'
    model_params['input_size'] = 224
    model_params['num_classes'] = 4
    return model_params


def get_lr_params(model_params, args=None):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type or 'wideresnet' in network_type:
        model_params['weight_decay'] = 0.0005

    else:
        model_params['weight_decay'] = 0.0001

    model_params['learning_rate'] = 0.1
    if model_params['task'] == 'imagenet':
        model_params['epochs'] = 0
        model_params['milestones'] = []
    elif model_params['task'] == 'oct2017':
        model_params['learning_rate'] = 0.01
        model_params['epochs'] = 20
        model_params['milestones'] = [5, 10, 15]
    else:
        model_params['epochs'] = 100
        model_params['milestones'] = [35, 60, 85]
    model_params['gammas'] = [0.1, 0.1, 0.1]

    # SDN ic_only training params
    model_params['ic_only'] = {}
    if model_params['task'] == 'imagenet':
        model_params['ic_only']['epochs'] = 40
        model_params['ic_only'][
            'learning_rate'] = 1e-5 * args.lr_scaler  # lr for full network training after sdn modification
        model_params['ic_only']['milestones'] = [20, 30]
        model_params['ic_only']['gammas'] = [0.1, 0.1]
    elif model_params['task'] == 'oct2017':
        model_params['ic_only']['epochs'] = 20
        model_params['ic_only'][
            'learning_rate'] = 1e-5 * args.lr_scaler  # lr for full network training after sdn modification
        model_params['ic_only']['milestones'] = [5, 10, 15]
        model_params['ic_only']['gammas'] = [0.1, 0.1, 0.1]
    else:
        if model_params['task'] == 'tinyimagenet':
            model_params['ic_only']['milestones'] = [15, 40]
            model_params['ic_only']['gammas'] = [0.1, 0.1]
        else:
            model_params['ic_only']['milestones'] = [15]
            model_params['ic_only']['gammas'] = [0.1]
        model_params['ic_only']['epochs'] = 50
        model_params['ic_only']['learning_rate'] = 0.001  # lr for full network training after sdn modification


def save_model(args,
               model,
               model_params,
               models_path,
               model_name,
               epoch=-1,
               train_outputs=None,
               test_outputs=None,
               total_ops=None,
               total_params=None):
    print(f'Saving model to {models_path}, epoch {epoch}')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    network_path = models_path + '/' + model_name

    if not os.path.exists(network_path):
        os.makedirs(network_path)

    # epoch == 0 is the untrained network, epoch == -1 is the last
    if epoch == 0:
        path = network_path + '/untrained'
        params_path = network_path + '/parameters_untrained'
    elif epoch == -1:
        path = network_path + '/last'
        params_path = network_path + '/parameters_last'
    else:
        path = network_path + '/' + str(epoch)
        params_path = network_path + '/parameters_' + str(epoch)

    print(f'Saving model to {path}')
    torch.save(model.state_dict(), path)

    if model_params is not None:
        print(f'Saving model params to {params_path}')
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)

    if total_ops is not None:
        torch.save(total_ops, f'{network_path}/total_ops.pt')
        args.run['artifacts/total_ops'].upload(f'{network_path}/total_ops.pt')

    if total_params is not None:
        torch.save(total_params, f'{network_path}/total_params.pt')
        args.run['artifacts/total_params'].upload(f'{network_path}/total_params.pt')

    if train_outputs is not None:
        logits, last_logits, labels = train_outputs
        print(f'Saving train logits to local filesystem...')
        torch.save(logits, f'{network_path}/train_logits.pt')
        torch.save(last_logits, f'{network_path}/train_last_logits.pt')
        torch.save(labels, f'{network_path}/train_labels.pt')
        if args.save_train_logits:
            print(f'Saving train logits to neptune...')
            args.run['artifacts/train_logits'].upload(f'{network_path}/train_logits.pt')
            args.run['artifacts/train_last_logits'].upload(f'{network_path}/train_last_logits.pt')
            args.run['artifacts/train_labels'].upload(f'{network_path}/train_labels.pt')
            neptune.log_artifact(f'{network_path}/train_logits.pt', 'train_logits')
            neptune.log_artifact(f'{network_path}/train_last_logits.pt', 'train_last_logits')
            neptune.log_artifact(f'{network_path}/train_labels.pt', 'train_labels')

    if test_outputs is not None:
        logits, last_logits, labels = test_outputs
        print(f'Saving test logits to local filesystem...')
        torch.save(logits, f'{network_path}/test_logits.pt')
        torch.save(last_logits, f'{network_path}/test_last_logits.pt')
        torch.save(labels, f'{network_path}/test_labels.pt')
        if args.save_test_logits:
            print(f'Saving test logits to neptune...')
            args.run['artifacts/test_logits'].upload(f'{network_path}/test_logits.pt')
            args.run['artifacts/test_last_logits'].upload(f'{network_path}/test_last_logits.pt')
            args.run['artifacts/test_labels'].upload(f'{network_path}/test_labels.pt')

    if hasattr(model, 'classifier_weights'):
        classifier_weights = model.classifier_weights
        print(f'Saving boosting weights to local filesystem...')
        torch.save(classifier_weights, f'{network_path}/boosting_weights.pt')
        print(f'Saving boosting weights to neptune...')
        args.run['artifacts/boosting_weights'].upload(f'{network_path}/boosting_weights.pt')

    print(f'Model saved')


def load_params(models_path, model_name, epoch=0):
    params_path = models_path + '/' + model_name
    if epoch == 0:
        params_path = params_path + '/parameters_untrained'
    else:
        params_path = params_path + '/parameters_last'

    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params


def load_model(args, models_path, model_name, epoch=0):
    model_params = load_params(models_path, model_name, epoch)

    architecture = 'empty' if 'architecture' not in model_params else model_params['architecture']
    network_type = model_params['network_type']

    if architecture == 'WeightedAverage':
        model = WeightedAverage(args, model_params)
    elif architecture == 'sdn' or 'sdn' in model_name:

        if 'wideresnet' in network_type:
            model = WideResNet_SDN(args, model_params)
        elif 'tv_resnet' in network_type:
            model = ResNet50_SDN(args, model_params)
        elif 'resnet' in network_type:
            model = ResNet_SDN(args, model_params)
        elif 'vgg' in network_type:
            model = VGG_SDN(args, model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(args, model_params)
    elif architecture == 'cnn' or 'cnn' in model_name:
        if 'wideresnet' in network_type:
            model = WideResNet(args, model_params)
        elif 'tv_resnet' in network_type:
            model = ResNet50(args, model_params)
        elif 'resnet' in network_type:
            model = ResNet(args, model_params)
        elif 'vgg' in network_type:
            model = VGG(args, model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(args, model_params)

    network_path = models_path + '/' + model_name

    if epoch == 0:  # untrained model
        load_path = network_path + '/untrained'
    elif epoch == -1:  # last model
        load_path = network_path + '/last'
    else:
        load_path = network_path + '/' + str(epoch)

    model.load_state_dict(torch.load(load_path), strict=False)

    return model, model_params


def get_sdn(cnn):
    if (isinstance(cnn, VGG)):
        return VGG_SDN
    elif (isinstance(cnn, ResNet)):
        return ResNet_SDN
    elif (isinstance(cnn, WideResNet)):
        return WideResNet_SDN
    elif (isinstance(cnn, MobileNet)):
        return MobileNet_SDN
    elif isinstance(cnn, ResNet50):
        return ResNet50_SDN


def get_cnn(sdn):
    if (isinstance(sdn, VGG_SDN)):
        return VGG
    elif (isinstance(sdn, ResNet_SDN)):
        return ResNet
    elif (isinstance(sdn, WideResNet_SDN)):
        return WideResNet
    elif (isinstance(sdn, MobileNet_SDN)):
        return MobileNet
    elif isinstance(sdn, ResNet50_SDN):
        return ResNet50


def get_net_params(args, net_type, task):
    if net_type == 'vgg16bn':
        return create_vgg16bn(args, None, task, None, True)
    elif net_type == 'resnet56':
        return create_resnet56(args, None, task, None, True)
    elif net_type == 'wideresnet32_4':
        return create_wideresnet32_4(args, None, task, None, True)
    elif net_type == 'mobilenet':
        return create_mobilenet(args, None, task, None, True)
    elif net_type == 'tv_resnet':
        return create_tv_resnet(args, None, task, None, True)

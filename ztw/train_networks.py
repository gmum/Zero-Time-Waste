# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function

import argparse
import copy
import os

import neptune.new as neptune
import torch

import aux_funcs as af
import network_architectures as arcs
from architectures.CNNs.VGG import VGG
from architectures.weighted_avg_model import WeightedAverage
from profiler import profile, profile_sdn


def get_logits(args, model, loader, device='cpu'):
    model.eval()
    all_logits = []
    all_last_logits = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            # y = y.to(device)
            output = model(X)
            if isinstance(output, torch.Tensor):
                all_logits.append(output.detach().cpu())
                all_last_logits.append(output.detach().cpu())
            else:
                stacked_output = torch.stack([torch.stack(o, 1) for o in output[:-1]], 1)
                all_logits.append(stacked_output.detach().cpu())
                all_last_logits.append(output[-1].detach().cpu())
            all_labels.append(y.detach().cpu())
    all_logits = torch.cat(all_logits, 0)
    all_last_logits = torch.cat(all_last_logits, 0)
    all_labels = torch.cat(all_labels, 0)
    model.train()
    return all_logits, all_last_logits, all_labels


def train(args, models_path, untrained_models, sdn=False, run_ensb=False, ic_only_sdn=False, tags=[], device='cpu'):
    print('Training models...')

    for base_model in untrained_models:
        trained_model, model_params = arcs.load_model(args, models_path, base_model, 0)
        dataset = af.get_dataset(args, model_params['task'])

        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = model_params['epochs']

        model_params['optimizer'] = 'SGD'

        if ic_only_sdn:  # IC-only training, freeze the original weights
            learning_rate = model_params['ic_only']['learning_rate']
            num_epochs = model_params['ic_only']['epochs']
            milestones = model_params['ic_only']['milestones']
            gammas = model_params['ic_only']['gammas']

            model_params['optimizer'] = 'Adam'
            trained_model.ic_only = True
        else:
            trained_model.ic_only = False

        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        trained_model_name = base_model

        # Add neptune tag for easier experiment management
        if args.tag:
            tags += [args.tag]

        print('Training: {}...'.format(trained_model_name))
        run = neptune.init(name=trained_model_name,
                           source_files='*.py',
                           tags=tags)
        run['parameters'] = vars(args)
        args.run = run
        trained_model.to(device)
        results = trained_model.train_func(args,
                                           trained_model,
                                           dataset,
                                           num_epochs,
                                           optimization_params,
                                           lr_schedule_params,
                                           device=device)
        if run_ensb:
            heads_weights = results['weights'].view(1, -1, 1)
            # TODO: fix this
            train_logits = trained_model.train_logits
            test_logits = trained_model.test_logits
        else:
            model_params['train_top1_acc'] = results['train_top1_acc']
            model_params['test_top1_acc'] = results['test_top1_acc']
            model_params['train_top5_acc'] = results['train_top5_acc']
            model_params['test_top5_acc'] = results['test_top5_acc']
            model_params['lrs'] = results['lrs']
            if not args.skip_train_logits:
                train_logits = get_logits(args, trained_model, dataset.eval_train_loader, device=device)
            else:
                train_logits = None
            test_logits = get_logits(args, trained_model, dataset.test_loader, device=device)
            example_x = next(iter(dataset.test_loader))

        if sdn:
            output_total_ops, output_total_params = profile_sdn(trained_model, example_x[0].size(2), device=device)
        elif not run_ensb:
            output_total_ops, output_total_params = profile(trained_model, example_x[0].size(2), device=device)
        else:
            output_total_ops, output_total_params = None, None


        model_params['epoch_times'] = results['epoch_times']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))

        arcs.save_model(args,
                        trained_model,
                        model_params,
                        models_path,
                        trained_model_name,
                        epoch=-1,
                        train_outputs=train_logits,
                        test_outputs=test_logits,
                        total_ops=output_total_ops,
                        total_params=output_total_params)


def train_sdns(args, models_path, networks, ic_only=False, device='cpu'):
    if ic_only:  # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else:  # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    for sdn_name in networks:
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_params = arcs.get_net_params(args, sdn_params['network_type'], sdn_params['task'])
        cnn_to_tune = f"{sdn_params['task']}_{sdn_params['network_type']}_cnn"
        sdn_model, _ = af.cnn_to_sdn(args, models_path, cnn_to_tune, sdn_params,
                                     load_epoch)  # load the CNN and convert it to a SDN
        arcs.save_model(args, sdn_model, sdn_params, models_path, sdn_name, epoch=0)  # save the resulting SDN
    train(args, models_path, networks, sdn=True, ic_only_sdn=ic_only, device=device)


def train_run_ensb(args, models_path, networks, ic_only=False, device='cpu'):
    if ic_only:  # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else:  # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    untrained_models = []
    for model_name in networks:
        model_params = arcs.load_params(models_path, model_name, epoch=load_epoch)
        model_params = arcs.get_net_params(args, model_params['network_type'], model_params['task'])
        dataset = af.get_dataset(args, model_params['task'])
        orig_model_name = model_name
        print("ORIG MODEL NAME", orig_model_name)

        base_model, _ = arcs.load_model(args, models_path, model_name, epoch=load_epoch)
        base_model.to(device)
        base_model.eval()

        # train_loader = dataset.eval_train_loader if args.dataset != "imagenet" else dataset.subset_train_loader
        # train_loader = dataset.eval_train_loader
        train_loader = dataset.eval_aug_train_loader
        # train_loader = dataset.subset_train_loader

        train_logits, train_last_logits, train_labels = get_logits(args, base_model,
                                                                   train_loader, device=device)
        test_logits, test_last_logits, test_labels = get_logits(args, base_model,
                                                                dataset.test_loader, device=device)

        for head_idx in args.head_ids:
            model_name = f"{orig_model_name}_add{args.run_ensb_type}_rensb_head_{head_idx}"
            model_params = copy.deepcopy(model_params)

            input_dim = (head_idx + 1) * model_params['num_classes']
            out_dim = model_params['num_classes']

            model_params['input_dim'] = input_dim
            model_params['output_dim'] = out_dim
            model_params['train_logits'] = train_logits
            model_params['train_last_logits'] = train_last_logits
            model_params['train_labels'] = train_labels
            model_params['test_logits'] = test_logits
            model_params['test_last_logits'] = test_last_logits
            model_params['test_labels'] = test_labels
            model_params['head_idx'] = head_idx
            model_params['input_type'] = 'probs' if args.run_ensb_type != 'geometric' else 'log_probs'
            model_params['softmax'] = False
            model_params['architecture'] = 'WeightedAverage'
            model_params['ensemble_mode'] = args.run_ensb_type
            model_params['ic_only']['learning_rate'] = 5e-4
            model_params['ic_only']['epochs'] = args.run_ensb_epochs

            model = WeightedAverage(args, model_params)
            model.to(device)

            arcs.save_model(args, model, model_params, models_path, model_name, epoch=0)  # save the resulting SDN
            untrained_models.append(model_name)

    tags = None
    if args.parent_id:
        tags = [args.parent_id]

    train(args, models_path,
          untrained_models=untrained_models, sdn=False,
          run_ensb=True, ic_only_sdn=ic_only, device=device, tags=tags)


def train_models(args, steps, task, models, models_path, device='cpu'):
    cnns = []
    sdns = []

    save_type = 'cd' if args.head_arch is not None else 'c'

    if 'vgg16bn' in models:
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(args, models_path, task, save_type=save_type))
    if 'resnet56' in models:
        af.extend_lists(cnns, sdns, arcs.create_resnet56(args, models_path, task, save_type=save_type))
    if 'wideresnet32_4' in models:
        af.extend_lists(cnns, sdns, arcs.create_wideresnet32_4(args, models_path, task, save_type=save_type))
    if 'mobilenet' in models:
        af.extend_lists(cnns, sdns, arcs.create_mobilenet(args, models_path, task, save_type=save_type))
    if 'tv_resnet' in models:
        af.extend_lists(cnns, sdns, arcs.create_tv_resnet(args, models_path, task, save_type=save_type))

    if 'cnn' in steps:
        train(args, models_path, cnns, sdn=False, device=device)
    if 'sdn_ic' in steps:
        train_sdns(args, models_path, sdns, ic_only=True, device=device)  # train SDNs with IC-only strategy
    if 'sdn_full' in steps:
        train_sdns(args, models_path, sdns, ic_only=False, device=device)  # train SDNs with SDN-training strategy
    if 'running_ensb' in steps:
        train_run_ensb(args, models_path, sdns, ic_only=True, device=device)


# for backdoored models, load a backdoored CNN and convert it to an SDN via IC-only strategy
def sdn_ic_only_backdoored(args, device):
    params = arcs.create_vgg16bn(None, 'cifar10', None, True)

    path = 'backdoored_models'
    backdoored_cnn_name = 'VGG16_cifar10_backdoored'
    save_sdn_name = 'VGG16_cifar10_backdoored_SDN'

    # Use the class VGG
    backdoored_cnn = VGG(params)
    backdoored_cnn.load_state_dict(torch.load('{}/{}'.format(path, backdoored_cnn_name), map_location='cpu'),
                                   strict=False)

    # convert backdoored cnn into a sdn
    backdoored_sdn, sdn_params = af.cnn_to_sdn(None, backdoored_cnn, params,
                                               preloaded=backdoored_cnn)  # load the CNN and convert it to a sdn
    arcs.save_model(args, backdoored_sdn, sdn_params, path, save_sdn_name, epoch=0)  # save the resulting sdn

    networks = [save_sdn_name]

    train(args, path, networks, sdn=True, ic_only_sdn=True, device=device)


def main(args):
    random_seed = args.seed
    af.set_rng_seeds(random_seed)
    print('RNG Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}'.format(random_seed)
    af.create_path(models_path)
    trained_models_dir = 'outputs/train_models'
    af.create_path(trained_models_dir)
    af.set_logger((os.path.join(trained_models_dir, str(random_seed))))

    train_models(args, args.training_steps, args.dataset, args.arch, models_path, device)

    # Commented out since it is not important right now
    # sdn_ic_only_backdoored(args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str,
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet', 'oct2017'])
    parser.add_argument('--arch',
                        '-a',
                        type=str,
                        choices=['vgg16bn', 'resnet56', 'wideresnet32_4', 'mobilenet', 'tv_resnet'])
    parser.add_argument('--training_steps',
                        '-t',
                        type=str,
                        nargs='+',
                        choices=['cnn', 'sdn_ic', 'sdn_full', 'running_ensb'],
                        default=['cnn', 'sdn_ic', 'sdn_full', 'running_ensb'])
    parser.add_argument('--head_arch', type=str, nargs='*',
                        choices=['conv', 'conv_less_ch', 'avg_pool', 'max_pool', 'sdn_pool', 'fc'])
    parser.add_argument('--sequential_training', action='store_true', help='train each ensemble member sequentially')
    parser.add_argument('--heads_per_ensemble', '-e', type=int, default=1)
    parser.add_argument('--seed', '-s', type=int)
    parser.add_argument('--size_after_pool', '-p', type=int)
    parser.add_argument('--loss',
                        type=str,
                        default='ce',
                        choices=['ce', 'bcewl', 'ce_kd', 'ce_auroc'],
                        help='criterion to use when training heads')
    parser.add_argument('--beta', type=float)
    parser.add_argument('--temperature', type=float, default=1., help='soft labels softmax temperature for KD')
    parser.add_argument('--lamb', type=float, default=0.75, help='parameter weighting KD hard and soft label losses')
    parser.add_argument('--boosting',
                        type=str,
                        default='off',
                        choices=['off', 'samme', 'sammer', 'confidence', 'confidence_sq'],
                        help='boosting / sequential head learning type')
    parser.add_argument('--stacking',
                        action='store_true')
    parser.add_argument('--detach_prev', action='store_true')
    parser.add_argument('--save_test_logits', action='store_true')
    parser.add_argument('--save_train_logits', action='store_true')
    parser.add_argument('--skip_train_logits', action='store_true')
    parser.add_argument('--alpha', type=float, help='alpha scaler for weight running ensembles', default=0.)
    parser.add_argument('--parent_id', type=str, help='id of neptune parent experiment')
    parser.add_argument('--heads', type=str, default='original',
                        choices=['all', 'original', 'half', 'third', 'quarter'])
    parser.add_argument('--detach_norm', type=str, default=None, choices=['layernorm'])
    parser.add_argument('--head_ids', type=int, nargs="+", help='head number to train with running ensembles')
    parser.add_argument('--examples_num', type=int, help='How many examples from the dataset to use')
    parser.add_argument('--run_ensb_epochs', type=int, default=501)
    parser.add_argument('--run_ensb_dataset', type=str, default='train')
    parser.add_argument('--run_ensb_type', choices=['geometric', 'additive', 'standard'], default='geometric',
                        help='ensemble type')
    parser.add_argument('--validation_dataset', action='store_true',
                        help='Use validation dataset for training (for running ensembles)')
    parser.add_argument('--head_shift', type=int, default=0)
    parser.add_argument('--lr_scaler', type=float, default=1.)
    parser.add_argument('--tag', type=str, help="Additional tag for neptune")
    parser.add_argument('--suffix', type=str, help="Suffix for model name")
    parser.add_argument('--relearn_final_layer', action='store_true')
    args = parser.parse_args()
    main(args)

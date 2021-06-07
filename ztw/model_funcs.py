# model_funcs.py
# implements the functions for training, testing SDNs and CNNs
# also implements the functions for computing confusion and confidence

import copy
import math
import random
import time
from collections import Counter
from random import choice, shuffle

import neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD

import aux_funcs as af
import data

from train_networks import get_logits

# fast ugly hack
current_step = 0


def sdn_training_step(args, optimizer, model, coeffs, batch, device, run_i=None):
    global current_step
    assert not args.loss == 'ce_kd', 'Distillation not implemented for whole-model head training'
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    criterion = CrossEntropyLoss()

    for ensemble_id in range(model.num_output - 1):
        for cur_output in output[ensemble_id]:
            cur_loss = float(coeffs[ensemble_id]) * criterion(cur_output, b_y)
            total_loss += cur_loss

    total_loss += criterion(output[-1], b_y)
    neptune.log_metric(f'loss run {run_i}', current_step, total_loss)
    total_loss.backward()
    optimizer.step()  # apply gradients
    current_step += 1

    return total_loss


def sdn_ic_only_step(args, optimizer, model, batch, device, run_i=None):
    global current_step
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    if args.loss == 'ce_kd':
        last_output = output[-1]
        q = F.softmax(last_output / args.temperature, dim=1)

    for ensemble_id, cur_ensemble in enumerate(output):
        if ensemble_id == model.num_output - 1:  # last output
            break
        for output_id, cur_output in enumerate(cur_ensemble):
            if args.loss == 'ce_kd':
                p = F.log_softmax(cur_output / args.temperature, dim=1)
                l_kl = F.kl_div(p, q, size_average=False) * args.temperature**2 / last_output.size(0)
                l_ce = F.cross_entropy(cur_output, b_y)
                neptune.log_metric(f'run {run_i} ens {ensemble_id} net {output_id} KL loss', current_step, l_kl)
                neptune.log_metric(f'run {run_i} ens {ensemble_id} net {output_id} CE loss', current_step, l_ce)
                l_kd = l_kl * args.lamb + l_ce * (1 - args.lamb)
                total_loss += l_kd
            elif args.loss == 'bcewl':
                cur_loss = F.binary_cross_entropy_with_logits(cur_output,
                                                              F.one_hot(b_y, num_classes=model.num_classes).float())
                total_loss += cur_loss
            elif args.loss == 'ce_auroc':
                y_max_pred = cur_output.exp().sum(dim=-1).log()
                y_pred_answers = cur_output.detach().argmax(dim=1)
                y_correct = y_pred_answers == b_y
                positive_activations = y_max_pred[y_correct]
                negative_activations = y_max_pred[~y_correct]
                diff = positive_activations.view(-1, 1) - negative_activations.view(1, -1)
                ca_auroc_loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
                ce_loss = F.cross_entropy(cur_output, b_y)
                cur_loss = (1 - args.beta) * ce_loss + (args.beta) * ca_auroc_loss
                total_loss += cur_loss
            else:
                cur_loss = F.cross_entropy(cur_output, b_y)
                total_loss += cur_loss

    neptune.log_metric(f'run {run_i} loss', current_step, total_loss)
    total_loss.backward()
    optimizer.step()  # apply gradients
    current_step += 1

    return total_loss


def sdn_boosting_step(args, optimizer, model, batch, device, head_i=None):
    global current_step
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    if args.loss == 'ce_kd':
        last_output = output[-1]
        q = F.softmax(last_output / args.temperature, dim=1)

    for ensemble_id, cur_ensemble in enumerate(output):
        if ensemble_id != head_i:
            continue
        for output_id, cur_output in enumerate(cur_ensemble):
            if args.loss == 'ce_kd':
                p = F.log_softmax(cur_output / args.temperature, dim=1)
                l_kl = F.kl_div(p, q, size_average=False) * args.temperature**2 / last_output.size(0)
                l_ce = F.cross_entropy(cur_output, b_y)
                neptune.log_metric(f'head {ensemble_id} net {output_id} KL loss', current_step, l_kl)
                neptune.log_metric(f'head {ensemble_id} net {output_id} CE loss', current_step, l_ce)
                l_kd = l_kl * args.lamb + l_ce * (1 - args.lamb)
                total_loss += l_kd
            elif args.loss == 'bcewl':
                cur_loss = F.binary_cross_entropy_with_logits(cur_output,
                                                              F.one_hot(b_y, num_classes=model.num_classes).float())
                total_loss += cur_loss
            elif args.loss == 'ce_auroc':
                y_max_pred = cur_output.exp().sum(dim=-1).log()
                y_pred_answers = cur_output.detach().argmax(dim=1)
                y_correct = y_pred_answers == b_y
                positive_activations = y_max_pred[y_correct]
                negative_activations = y_max_pred[~y_correct]
                diff = positive_activations.view(-1, 1) - negative_activations.view(1, -1)
                ca_auroc_loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
                ce_loss = F.cross_entropy(cur_output, b_y)
                cur_loss = (1 - args.beta) * ce_loss + (args.beta) * ca_auroc_loss
                total_loss += cur_loss
            else:
                cur_loss = F.cross_entropy(cur_output, b_y)
                total_loss += cur_loss

    neptune.log_metric(f'head {head_i} loss', current_step, total_loss)
    total_loss.backward()
    optimizer.step()  # apply gradients
    current_step += 1

    return total_loss


def get_loader(data, augment):
    if augment:
        train_loader = data.aug_train_loader
    else:
        train_loader = data.train_loader

    return train_loader


def samme_error(model, data, weights, head_i, device):
    weights = weights.to(device)
    model.eval()
    loader = data.eval_train_loader
    idx = 0
    weighted_error = 0.0
    for batch in loader:
        b_x = batch[0].to(device)
        b_y = batch[1].to(device)
        batch_size = b_x.size(0)
        output = model(b_x)
        for ensemble_id, cur_ensemble in enumerate(output):
            if ensemble_id != head_i:
                continue
            # TODO support ensembles?
            assert len(cur_ensemble) == 1
            head_answers = cur_ensemble[0].argmax(dim=-1)
            weighted_error += ((head_answers != b_y).float() * weights[idx:idx + batch_size]).sum(dim=0)
        idx += batch_size
    weighted_error /= weights.sum()
    model.train()
    return weighted_error


def samme_classifier_weight(err, num_classes):
    return math.log((1 - err) / err) + math.log(num_classes - 1)


def samme_sampling_weights(weights, classifier_weight, model, data, head_i, device):
    weights = weights.to(device)
    model.eval()
    loader = data.eval_train_loader
    new_weights = torch.zeros_like(weights)
    idx = 0
    for batch in loader:
        b_x = batch[0].to(device)
        b_y = batch[1].to(device)
        batch_size = b_x.size(0)
        output = model(b_x)
        for ensemble_id, cur_ensemble in enumerate(output):
            if ensemble_id != head_i:
                continue
            # TODO support ensembles?
            assert len(cur_ensemble) == 1
            head_logits = cur_ensemble[0]
            head_answers = head_logits.argmax(dim=-1)
            err_indicator = (head_answers != b_y).float()
            new_weights[idx:idx +
                        batch_size] = weights[idx:idx + batch_size] * torch.exp(classifier_weight * err_indicator)
        idx += batch_size
    # and renormalize
    new_weights /= new_weights.sum()
    model.train()
    return new_weights


def sammer_sampling_weights(weights, model, data, head_i, device):
    weights = weights.to(device)
    model.eval()
    loader = data.eval_train_loader
    new_weights = torch.zeros_like(weights)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            batch_size = b_x.size(0)
            output = model(b_x)
            for ensemble_id, cur_ensemble in enumerate(output):
                if ensemble_id != head_i:
                    continue
                # TODO support ensembles?
                assert len(cur_ensemble) == 1
                head_probs = torch.softmax(cur_ensemble[0], dim=-1)
                # sum_zero_b_y = -torch.ones_like(head_probs) / (model.num_classes - 1)
                sum_zero_b_y = -torch.div(torch.ones_like(head_probs), float(model.num_classes - 1))
                sum_zero_b_y.scatter_(1, b_y.unsqueeze(1), 1)
                exponent = -(model.num_classes - 1) / model.num_classes * (sum_zero_b_y * head_probs.log()).sum(dim=-1)
                new_weights[idx:idx + batch_size] = weights[idx:idx + batch_size] * torch.exp(exponent).detach()
            idx += batch_size
    # and renormalize
    new_weights /= new_weights.sum()
    model.train()
    return new_weights.detach()


def confidence_weights(args, model, data, head_i, device):
    # calculates weights for head head_i
    model.eval()
    loader = data.eval_train_loader
    new_weights = torch.zeros(len(data.trainset), head_i + 1, device=device)
    idx = 0
    for batch in loader:
        b_x = batch[0].to(device)
        b_y = batch[1].to(device)
        batch_size = b_x.size(0)
        output = model(b_x)
        for ensemble_id, cur_ensemble in enumerate(output):
            if ensemble_id <= head_i:
                # TODO support ensembles?
                assert len(cur_ensemble) == 1
                head_confidence = torch.softmax(cur_ensemble[0], dim=-1).max(dim=-1)[0]
                if args.boosting == 'confidence':
                    new_weights[idx:idx + batch_size, head_i] = (1 - head_confidence).detach().cpu()
                elif args.boosting == 'confidence_sq':
                    new_weights[idx:idx + batch_size, head_i] = ((1 - head_confidence)**2).detach().cpu()
        idx += batch_size
    model.train()
    if args.conf_reduction == 'max':
        return new_weights.max(dim=-1)[0]
    elif args.conf_reduction == 'mean':
        return new_weights.mean(dim=-1)


def sdn_train(args, model, data, epochs, optimization_params, lr_schedule_params, device='cpu'):
    global current_step
    print(model)
    metrics = {
        'epoch_times': [],
        'test_top1_acc': [],
        'test_top5_acc': [],
        'train_top1_acc': [],
        'train_top5_acc': [],
        'lrs': []
    }
    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')
    if args.boosting != 'off':
        if args.boosting == 'samme':
            classifier_weights = torch.zeros(model.num_output - 1)
        data_len = len(data.trainset)
        dataset_weights = torch.ones(data_len) / data_len
        data.weighted_loaders(dataset_weights)
        for i_head in range(model.num_output - 1):
            current_step = 0
            augment = model.augment_training
            af.freeze(model, i_head, boosting=True)
            optimizer, scheduler = af.get_sdn_ic_only_optimizer(model, optimization_params, lr_schedule_params)
            for epoch in range(1, epochs + 1):
                cur_lr = af.get_lr(optimizer)
                print('\nEpoch: {}/{}'.format(epoch, epochs))
                print('Cur lr: {}'.format(cur_lr))
                neptune.log_metric(f'boosting classifier {i_head} learning rate', current_step, cur_lr)

                start_time = time.time()
                model.train()
                loader = get_loader(data, augment)
                for i, batch in enumerate(loader):
                    total_loss = sdn_boosting_step(args, optimizer, model, batch, device, i_head)
                    if i % 100 == 0:
                        print('Loss: {}: '.format(total_loss))
                scheduler.step()

                top1_test, top5_test = sdn_test(model, data.test_loader, device)
                for i, test_acc in enumerate(top1_test):
                    neptune.log_metric(f'run for head {i_head} test acc {i}', current_step, test_acc)
                print('Top1 Test accuracies: {}'.format(top1_test))
                print('Top5 Test accuracies: {}'.format(top5_test))
                end_time = time.time()

                metrics['test_top1_acc'].append(top1_test)
                metrics['test_top5_acc'].append(top5_test)

                top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
                for i, train_acc in enumerate(top1_train):
                    neptune.log_metric(f'run for head {i_head} train acc {i} ', current_step, train_acc)
                print('Top1 Train accuracies: {}'.format(top1_train))
                print('Top5 Train accuracies: {}'.format(top5_train))
                metrics['train_top1_acc'].append(top1_train)
                metrics['train_top5_acc'].append(top5_train)
                
                epoch_time = int(end_time - start_time)
                metrics['epoch_times'].append(epoch_time)
                print('Epoch took {} seconds.'.format(epoch_time))

                metrics['lrs'].append(cur_lr)

            if args.boosting == 'samme':
                err = samme_error(model, data, dataset_weights, i_head, device)
                classifier_weights[i_head] = samme_classifier_weight(err, data.num_classes)
                dataset_weights = samme_sampling_weights(dataset_weights, classifier_weights[i_head], model, data,
                                                         i_head, device)
            elif args.boosting == 'sammer':
                dataset_weights = sammer_sampling_weights(dataset_weights, model, data, i_head, device)
            elif 'confidence' in args.boosting:
                dataset_weights = confidence_weights(args, model, data, i_head, device)
            data.weighted_loaders(dataset_weights)
        if args.boosting == 'samme':
            model.classifier_weights = classifier_weights
    else:
        n_train = args.heads_per_ensemble if args.sequential_training else 1
        for i_train in range(n_train):
            current_step = 0
            if model.ic_only:
                if args.sequential_training:
                    af.freeze(model, i_train)
                else:
                    af.freeze(model, 'except_outputs')
                optimizer, scheduler = af.get_sdn_ic_only_optimizer(model, optimization_params, lr_schedule_params)
            else:
                optimizer, scheduler = af.get_full_optimizer(model, optimization_params, lr_schedule_params)
                af.freeze(model, 'nothing')
            augment = model.augment_training
            max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values

            for epoch in range(1, epochs + 1):
                cur_lr = af.get_lr(optimizer)
                print('\nEpoch: {}/{}'.format(epoch, epochs))
                print('Cur lr: {}'.format(cur_lr))
                neptune.log_metric(f'run {i_train} learning rate', current_step, cur_lr)

                if model.ic_only is False:
                    # calculate the IC coeffs for this epoch for the weighted objective function
                    cur_coeffs = 0.01 + epoch * (max_coeffs / epochs)  # to calculate the tau at the currect epoch
                    cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
                    print('Cur coeffs: {}'.format(cur_coeffs))

                start_time = time.time()
                model.train()
                loader = get_loader(data, augment)
                for i, batch in enumerate(loader):
                    if model.ic_only is False:
                        total_loss = sdn_training_step(args, optimizer, model, cur_coeffs, batch, device, run_i=i_train)
                    else:
                        total_loss = sdn_ic_only_step(args, optimizer, model, batch, device, run_i=i_train)

                    if i % 100 == 0:
                        print('Loss: {}: '.format(total_loss))
                scheduler.step()

                top1_test, top5_test = sdn_test(model, data.test_loader, device)
                for i, test_acc in enumerate(top1_test):
                    neptune.log_metric(f'run {i_train} test acc {i}', current_step, test_acc)
                print('Top1 Test accuracies: {}'.format(top1_test))
                print('Top5 Test accuracies: {}'.format(top5_test))
                end_time = time.time()

                metrics['test_top1_acc'].append(top1_test)
                metrics['test_top5_acc'].append(top5_test)

                top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
                for i, train_acc in enumerate(top1_train):
                    neptune.log_metric(f'run {i_train} train acc {i} ', current_step, train_acc)
                print('Top1 Train accuracies: {}'.format(top1_train))
                print('Top5 Train accuracies: {}'.format(top5_train))
                # metrics['train_top1_acc'].append(top1_train)
                # metrics['train_top5_acc'].append(top5_train)

                epoch_time = int(end_time - start_time)
                metrics['epoch_times'].append(epoch_time)
                print('Epoch took {} seconds.'.format(epoch_time))

                metrics['lrs'].append(cur_lr)

    return metrics


def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []

    for _ in range(model.num_output):
        t1 = data.AverageMeter()
        t5 = data.AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for ensemble_id in range(model.num_output):
                if ensemble_id == model.num_output - 1:  # Final layer
                    cur_output = output[ensemble_id]
                else:
                    cur_output = torch.stack(output[ensemble_id], 0)
                    cur_output = torch.softmax(cur_output, -1).mean(0)
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
                top1[ensemble_id].update(prec1[0], b_x.size(0))
                top5[ensemble_id].update(prec5[0], b_x.size(0))

    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    model.train()
    return top1_accs, top5_accs


def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id * loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_get_confusion(model, loader, confusion_stats, device='cpu', save_name=None):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, confusion_stats, device)

            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion


# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)


def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time


def cnn_training_step(args, model, optimizer, data, labels, device='cpu'):
    global current_step
    b_x = data.to(device)  # batch x
    b_y = labels.to(device)  # batch y
    output = model(b_x)  # cnn final output
    criterion = CrossEntropyLoss()
    loss = criterion(output, b_y)  # cross entropy loss
    neptune.log_metric('loss', current_step, loss)
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    current_step += 1


def cnn_train(args, model, data, epochs, optimization_params, lr_schedule_params, device='cpu'):
    global current_step
    current_step = 0
    metrics = {
        'epoch_times': [],
        'test_top1_acc': [],
        'test_top5_acc': [],
        'train_top1_acc': [],
        'train_top5_acc': [],
        'lrs': []
    }
    optimizer, scheduler = af.get_full_optimizer(model, optimization_params, lr_schedule_params)

    for epoch in range(1, epochs + 1):

        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        neptune.log_metric('learning rate', current_step, cur_lr)
        for x, y in train_loader:
            cnn_training_step(args, model, optimizer, x, y, device)
        scheduler.step()

        end_time = time.time()

        top1_test, top5_test = cnn_test(model, data.test_loader, device)
        neptune.log_metric('test acc', current_step, top1_test)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = cnn_test(model, train_loader, device)
        neptune.log_metric('train acc', current_step, top1_train)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('Top5 Train accuracy: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)
        epoch_time = int(end_time - start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)

    return metrics


def cnn_test_time(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, total_time


def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc


def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

    return correct, wrong, instance_confidence

def run_ensb_train(args, model, data, epochs, optimization_params, lr_schedule_params, device='cpu'):
    results = {
        'epoch_times': []
    }

    # Enforcing that the device is CPU
    train_logits = model.train_logits[:, :, 0].cpu()
    train_last_logits = model.train_last_logits.view(-1, 1, model.num_classes).cpu()

    train_logits = torch.cat([train_logits, train_last_logits], 1)
    train_labels = model.train_labels.cpu()

    test_logits = model.test_logits[:, :, 0].cpu()
    test_last_logits = model.test_last_logits.view(-1, 1, model.num_classes).cpu()

    test_logits = torch.cat([test_logits, test_last_logits], 1)
    test_labels = model.test_labels.cpu()

    model = model.to(device)

    if model.input_type == "logits":
        train_input = train_logits
        test_input = test_logits
    elif model.input_type == "probs":
        train_input = train_logits.softmax(-1)
        test_input = test_logits.softmax(-1)
    elif model.input_type == "log_probs":
        train_input = train_logits.log_softmax(-1)
        test_input = test_logits.log_softmax(-1)

    if args.run_ensb_dataset == "test":
        epochs = epochs * (len(train_input) // 1000)  # Keep the same number of updates
        train_input = test_input[:1000].clone()
        train_labels = test_labels[:1000].clone()
        print("Training running ensb on a subset of the test set!!!")

    if args.dataset == "imagenet":
        epochs = 10

    test_base_accs = (test_input.argmax(-1).squeeze() == test_labels.view(-1, 1)).float().mean(0).tolist()
    train_base_accs = (train_input.argmax(-1).squeeze() == train_labels.view(-1, 1)).float().mean(0).tolist()

    lr = optimization_params[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model.input_type != "probs":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.NLLLoss()

    train_dataset = torch.utils.data.TensorDataset(train_input, train_labels)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=512, shuffle=True)

    num_heads = train_logits.shape[1]


    for epoch in range(epochs):
        start_time = time.time()
        model.train()

        avg_loss = 0.
        for x, y in train_loader:
            x = x[:, :model.head_idx + 1].to(device)
            outputs = model(x)
            loss_val = loss_fn(outputs, y.to(device)) + args.alpha * torch.sum((model.weight - model.weight.mean()) ** 2)
            avg_loss += loss_val.cpu().item()
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            model = model.cpu()
            x = train_input[:, :model.head_idx + 1]
            scaled_preds = model(x).argmax(-1)
            acc = (scaled_preds == train_labels).float().mean()

            x = train_input[:, :model.head_idx + 1]
            test_inp = test_input[:, :model.head_idx + 1]
            scaled_preds = model(x).argmax(1)
            acc = (scaled_preds == train_labels).float().mean(0)

            test_scaled_preds = model(test_inp).argmax(1)
            test_acc = (test_scaled_preds == test_labels).float().mean(0)
            print(epoch, avg_loss / len(train_loader), test_acc)

            neptune.log_metric(f'head_{model.head_idx}_train_loss', epoch, avg_loss / len(train_loader))
            neptune.log_metric(f'head_{model.head_idx}_train_acc', epoch, acc.item())
            neptune.log_metric(f'head_{model.head_idx}_test_acc', epoch, test_acc.item())
            model = model.cuda()

        epoch_time = int(time.time() - start_time)
        results['epoch_times'].append(epoch_time)

    model = model.cpu()
    model.eval()
    print("Head", model.head_idx, "Base acc, ", train_base_accs[model.head_idx], "Train Ensembled acc", acc)
    print("Weights", model.weight)
    print("Bias", model.bias)

    new_train_logits = model(train_input[:, :model.head_idx + 1]).cpu()
    new_test_logits = model(test_input[:, :model.head_idx + 1]).cpu()

    model.train_logits = (new_train_logits, torch.tensor(0), train_labels.cpu())
    model.test_logits = (new_test_logits, torch.tensor(0), test_labels.cpu())

    print("Test Base accs, ", test_base_accs[model.head_idx], "Test Ensembled acc", test_acc)

    results['weights'] = model.weight
    results['bias'] = model.bias

    return results

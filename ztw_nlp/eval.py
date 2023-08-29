import logging
from typing import List, Tuple, Dict

import torch
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn import MultiheadAttention

from architectures.early_exits.pbee import PBEE
from utils import flop_count


def test_classification(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        batches: int = 0) -> Tuple[float, float, float]:
    criterion = criterion_class(reduction='sum')
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct, total = 0, 0
        preds = []
        labels = []
        for batch, (X, y) in enumerate(data_loader):
            y_pred = model(X)
            y_pred, y = accelerator.gather_for_metrics((y_pred, y))
            y_pred_max = y_pred.argmax(dim=1)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            preds.append(y_pred_max.detach().cpu())
            labels.append(y.detach().cpu())
            total += y.size(0)
            if batch >= batches > 0:
                break
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = (preds == labels).sum().item() / total
        f1 = f1_score(preds, labels, average='macro')
    # loss, acc, f1
    return running_loss / total, acc, f1


def get_preds(accelerator: Accelerator,
              model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            output = model(X)
            output, y = accelerator.gather_for_metrics((output, y))
            batch_outputs.append(output.detach().cpu())
            batch_labels.append(y.detach().cpu())
            if batch >= batches > 0:
                break
    preds = torch.cat(batch_outputs)
    labels = torch.cat(batch_labels)
    return preds, labels


def get_preds_earlyexiting(accelerator: Accelerator,
                           model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    with torch.no_grad():
        model.all_mode()
        for batch, (X, y) in enumerate(data_loader):
            output = model(X)
            output, y = accelerator.gather_for_metrics((output, y))
            y_preds = [y_pred.detach().cpu() for y_pred in output]
            batch_outputs.append(y_preds)
            batch_labels.append(y.detach().cpu())
            if batch >= batches > 0:
                break
    head_preds = []
    for i in range(model.number_of_heads):
        head_outputs = torch.cat([batch_output[i] for batch_output in batch_outputs])
        head_preds.append(head_outputs)
    labels = torch.cat(batch_labels)
    return head_preds, labels


def test_earlyexiting_classification(accelerator: Accelerator,
                                     model: torch.nn.Module,
                                     data_loader: torch.utils.data.DataLoader,
                                     criterion_class: torch.nn.Module,
                                     batches: int = 0):
    criterion = criterion_class(reduction='mean')
    head_preds, ys = get_preds_earlyexiting(accelerator, model, data_loader, batches)
    head_losses = [criterion(preds, ys) for preds in head_preds]
    head_accuracies = [(preds.argmax(dim=1) == ys).sum().item() / ys.size(0) for preds in head_preds]
    head_f1s = [f1_score(ys, preds.argmax(dim=1), average='macro') for preds in head_preds]
    return head_losses, head_accuracies, head_f1s


def average_earlyexiting_flops(head_costs: List, head_exit_counts: torch.Tensor):
    assert len(head_costs) == head_exit_counts.size(0), f'{head_costs=}\n{head_exit_counts=}'
    total_cost = 0.0
    for h_i, h_c in enumerate(head_costs):
        total_cost += h_c * head_exit_counts[h_i].item()
    average_cost = total_cost / head_exit_counts.sum().item()
    return average_cost


def evaluate_earlyexiting_classification(model: torch.nn.Module,
                                         head_preds: List[torch.Tensor],
                                         labels: torch.Tensor,
                                         head_costs: List[FlopCountAnalysis],
                                         eval_thresholds: int) -> Dict:
    head_accuracies = []
    head_f1s = []
    for i, head_pred in enumerate(head_preds):
        head_accuracy = (head_pred.argmax(dim=1) == labels).sum().item() / labels.size(0)
        head_accuracies.append(head_accuracy)
        head_f1 = f1_score(head_pred.argmax(dim=1).numpy(), labels.numpy(), average='macro')
        head_f1s.append(head_f1)
    head_flops = [head_cost.total() for head_cost in head_costs]
    thresholds = torch.linspace(0.0, 1.0, steps=eval_thresholds, device=labels.device)
    threshold_accuracies = []
    threshold_f1s = []
    threshold_flops = []
    # separate path for evaluating PBEE
    if isinstance(model, PBEE):
        patience_thresholds = torch.arange(0, len(head_preds), device=labels.device)
        for patience_threshold in patience_thresholds:
            exit_at = torch.zeros_like(labels) - 1
            outputs = torch.zeros_like(head_preds[0])
            # start from second head, set patience to one after first head
            prev_answers = torch.zeros_like(labels) - 1
            patience = torch.zeros_like(prev_answers)
            for i, head_pred in enumerate(head_preds):
                patience = torch.where(head_pred.argmax(-1) == prev_answers, patience + 1, 1)
                unresolved_mask = exit_at == -1
                exit_mask = (patience > patience_threshold) & unresolved_mask
                exit_at[exit_mask] = i
                outputs[exit_mask] = head_pred[exit_mask]
                prev_answers = head_pred.argmax(-1)
            unresolved_mask = exit_at == -1
            outputs[unresolved_mask] = head_preds[-1][unresolved_mask]
            exit_at[unresolved_mask] = len(head_preds) - 1
            threshold_accuracy = ((outputs.argmax(dim=-1) == labels).sum() / labels.size(0)).item()
            threshold_f1 = f1_score(outputs.argmax(dim=-1).numpy(), labels.numpy(), average='macro')
            exits_bincounted = exit_at.bincount(minlength=len(head_preds))
            threshold_cost = average_earlyexiting_flops(head_flops, exits_bincounted)
            threshold_accuracies.append(threshold_accuracy)
            threshold_f1s.append(threshold_f1)
            threshold_flops.append(threshold_cost)
    else:
        head_probs = [preds.softmax(dim=-1) for preds in head_preds]
        thresholds = torch.linspace(0.0, 1.0, steps=eval_thresholds, device=labels.device)
        for threshold in thresholds:
            exit_at = torch.zeros_like(labels) - 1
            outputs = torch.zeros_like(head_probs[0])
            for i, head_prob in enumerate(head_probs):
                head_confidences, _ = head_prob.max(dim=-1)
                unresolved_mask = exit_at == -1
                exit_mask = (head_confidences > threshold) & unresolved_mask
                exit_at[exit_mask] = i
                outputs[exit_mask] = head_prob[exit_mask]
            unresolved_mask = exit_at == -1
            outputs[unresolved_mask] = head_probs[-1][unresolved_mask]
            exit_at[unresolved_mask] = len(head_probs) - 1
            threshold_accuracy = ((outputs.argmax(dim=-1) == labels).sum() / labels.size(0)).item()
            threshold_f1 = f1_score(outputs.argmax(dim=-1).numpy(), labels.numpy(), average='macro')
            exits_bincounted = exit_at.bincount(minlength=len(head_probs))
            threshold_cost = average_earlyexiting_flops(head_flops, exits_bincounted)
            threshold_accuracies.append(threshold_accuracy)
            threshold_f1s.append(threshold_f1)
            threshold_flops.append(threshold_cost)
    results = {'head_scores': head_accuracies, 'head_f1s': head_f1s, 'head_flops': head_flops, 'thresholds': thresholds,
               'threshold_scores': threshold_accuracies, 'threshold_f1s': threshold_f1s,
               'threshold_flops': threshold_flops}
    return results


def evaluate_classification(preds: torch.Tensor, labels: torch.Tensor, criterion_class: torch.nn.Module):
    criterion = criterion_class(reduction='mean')
    preds_max = preds.argmax(dim=1)
    loss = criterion(preds, labels).item()
    accuracy = (preds_max == labels).double().mean().item()
    return loss, accuracy


def ks_calibration_error(probs, labels):
    '''https://arxiv.org/abs/2006.12800'''
    assert probs.dim() == 2, f'{probs.size()=}'
    num_classes = probs.size(-1)
    labels_oh = torch.nn.functional.one_hot(labels, num_classes)
    num_samples = probs.size(0)
    ks_errors = [0.0] * num_classes
    for k in range(num_classes):
        class_probs = probs[..., k]
        class_labels = labels_oh[..., k]
        sorted_probs, indices = class_probs.sort()
        h_tilde = torch.cumsum(class_labels[indices] / num_samples, dim=0)
        h = torch.cumsum(sorted_probs / num_samples, dim=0)
        ks_errors[k] += (h - h_tilde).abs().max().item()
    # TODO is averaging appropriate?
    ks_error = sum(ks_errors) / num_classes
    return ks_error, ks_errors


def evaluate_calibration(preds: torch.Tensor,
                         labels: torch.Tensor) -> Dict:
    probs = preds.softmax(dim=-1)
    # ignores per-class calibration scores, takes the average
    calibration_score, _ = ks_calibration_error(probs, labels)
    results = {'final_score': calibration_score}
    return results


# TODO possibly generalize this code and merge it with accuracy and ood
def evaluate_earlyexiting_calibration(head_preds: List[torch.Tensor],
                                      labels: torch.Tensor,
                                      head_costs: List[int],
                                      thresholds: torch.Tensor) -> Dict:
    head_probs = [preds.softmax(dim=-1) for preds in head_preds]
    head_calibration_scores = []
    for i, head_prob in enumerate(head_probs):
        head_calibration_score, _ = ks_calibration_error(head_prob, labels)
        head_calibration_scores.append(head_calibration_score)
    threshold_calibration_scores = []
    threshold_flops = []
    for threshold in thresholds:
        exit_at = torch.zeros_like(labels) - 1
        outputs = torch.zeros_like(head_probs[0])
        for i, head_prob in enumerate(head_probs):
            head_confidences, _ = head_prob.max(dim=-1)
            unresolved_mask = exit_at == -1
            exit_mask = (head_confidences > threshold) & unresolved_mask
            exit_at[exit_mask] = i
            outputs[exit_mask] = head_prob[exit_mask]
        unresolved_mask = exit_at == -1
        outputs[unresolved_mask] = head_probs[-1][unresolved_mask]
        exit_at[unresolved_mask] = len(head_probs) - 1
        threshold_calibration_score, _ = ks_calibration_error(outputs, labels)
        exits_bincounted = exit_at.bincount(minlength=len(head_probs))
        threshold_cost = average_earlyexiting_flops(head_costs, exits_bincounted)
        threshold_calibration_scores.append(threshold_calibration_score)
        threshold_flops.append(threshold_cost)
    results = {'head_scores': head_calibration_scores, 'head_flops': head_costs, 'thresholds': thresholds,
               'threshold_scores': threshold_calibration_scores, 'threshold_flops': threshold_flops}
    return results


def evaluate_ood_detection(id_preds: List[torch.Tensor],
                           ood_preds: torch.Tensor) -> Dict:
    id_confidences = id_preds.softmax(dim=-1).max(dim=-1)[0]
    ood_confidences = ood_preds.softmax(dim=-1).max(dim=-1)[0]
    confidences = torch.cat([id_confidences, ood_confidences])
    ood_labels = torch.cat([torch.ones_like(id_confidences), torch.zeros_like(ood_confidences)])
    ood_score = roc_auc_score(ood_labels.cpu().numpy(), confidences.cpu().numpy())
    assert 0.0 <= ood_score <= 1.0, f'AUROC: {ood_score}'
    results = {'final_score': ood_score}
    return results


def evaluate_earlyexiting_ood_detection(head_id_preds: List[torch.Tensor],
                                        head_ood_preds: List[torch.Tensor],
                                        head_costs: List[int],
                                        thresholds: torch.Tensor) -> Dict:
    # TODO this assumes the head costs are the same for the OOD dataset - add support for different costs
    head_id_confidences = [preds.softmax(dim=-1).max(dim=-1)[0] for preds in head_id_preds]
    head_ood_confidences = [preds.softmax(dim=-1).max(dim=-1)[0] for preds in head_ood_preds]
    head_confidences = [torch.cat([id_confidences, ood_confidences]) for id_confidences, ood_confidences in
                        zip(head_id_confidences, head_ood_confidences)]
    ood_labels = torch.cat([torch.ones_like(head_id_confidences[0], dtype=torch.int),
                            torch.zeros_like(head_ood_confidences[0], dtype=torch.int)])
    head_ood_scores = []
    for i, head_confs in enumerate(head_confidences):
        head_ood_score = roc_auc_score(ood_labels.cpu().numpy(), head_confs.cpu().numpy())
        head_ood_scores.append(head_ood_score)
    threshold_ood_scores = []
    threshold_flops = []
    for threshold in thresholds:
        exit_at = torch.zeros_like(ood_labels) - 1
        outputs = torch.zeros_like(head_confidences[0])
        for i, head_confs in enumerate(head_confidences):
            unresolved_mask = exit_at == -1
            exit_mask = (head_confs > threshold) & unresolved_mask
            exit_at[exit_mask] = i
            outputs[exit_mask] = head_confs[exit_mask]
        unresolved_mask = exit_at == -1
        outputs[unresolved_mask] = head_confidences[-1][unresolved_mask]
        exit_at[unresolved_mask] = len(head_confidences) - 1
        threshold_ood_detection_score = roc_auc_score(ood_labels.cpu().numpy(), outputs.cpu().numpy())
        exits_bincounted = exit_at.bincount(minlength=len(head_confidences))
        threshold_cost = average_earlyexiting_flops(head_costs, exits_bincounted)
        threshold_ood_scores.append(threshold_ood_detection_score)
        threshold_flops.append(threshold_cost)
    results = {'head_scores': head_ood_scores, 'head_flops': head_costs, 'thresholds': thresholds,
               'threshold_scores': threshold_ood_scores, 'threshold_flops': threshold_flops}
    return results


def benchmark(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader) -> Tuple[FlopCountAnalysis, Dict]:
    model.eval()
    # workaround for the missing implementation of 'aten::_native_multi_head_attention' flop counter
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.train()
    #
    X, _ = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[:1] for k, v in X.items()}
    else:
        sample = X[:1]

    with torch.inference_mode():
        model_costs = flop_count(model, (sample,))
        param_count = parameter_count(model)
    logging.info(f'Ops by operator:\n{model_costs.by_operator()}')
    logging.info(f'Ops by module:\n{flop_count_table(model_costs, max_depth=7)}')
    logging.info(f'Total ops: {model_costs.total()}')
    unsupported = model_costs.unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f'Unsupported op: {k} (occurrences: {v})')
    uncalled = model_costs.uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f'Uncalled module: {m}')
    return model_costs, param_count


def benchmark_earlyexiting(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader) \
        -> Tuple[List[FlopCountAnalysis], Dict]:
    model.eval()
    # workaround for the missing implementation of 'aten::_native_multi_head_attention' flop counter
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.train()
    #
    X, _ = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[0].unsqueeze(0) for k, v in X.items()}
    else:
        sample = X[0].unsqueeze(0)
    with torch.inference_mode():
        param_count = parameter_count(model)
        head_costs = []
        for head_i in range(model.number_of_heads):
            model.select_head(head_i)
            head_costs.append(flop_count(model, (sample,)))
            logging.info(f'Ops for head {head_i}: {head_costs[head_i].total()}')
    unsupported = head_costs[-1].unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f'Unsupported op: {k} (occurrences: {v})')
    uncalled = head_costs[-1].uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f'Uncalled module: {m}')
    return head_costs, param_count

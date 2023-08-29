import logging
import math
from copy import deepcopy
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from architectures.early_exits.l2w import TanhWPN
from common import INIT_NAME_MAP, get_default_args, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP
from methods.early_exit import set_for_training, final_eval, in_training_eval
from train import TrainingContext, setup_accelerator, setup_data, setup_files_and_logging, setup_state
from train import setup_optimization as standard_setup_optimization
from utils import save_state, load_model, get_lrs, create_model, Mixup, unfrozen_parameters, set_parameter_by_name


class L2wTrainingContext(TrainingContext):
    weight_prediction_network: torch.nn.Module = None
    wpn_optimizer: torch.optim.Optimizer = None
    wpn_scheduler: Any = None


def setup_model(args, tc):
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    model_args = {'base_model': base_model, **args.model_args}
    model = create_model(args.model_class, model_args)
    # setup wpn
    wpn_model = TanhWPN(model.number_of_heads, model.number_of_heads, args.wpn_width, args.wpn_depth)
    # init both networks
    init_fun = INIT_NAME_MAP[args.init_fun if args.init_fun is not None else base_args.init_fun]
    if init_fun is not None:
        for head_module in model.head_modules():
            init_fun(head_module)
        init_fun(wpn_model)
    tc.model = tc.accelerator.prepare(model)
    tc.weight_prediction_network = tc.accelerator.prepare(wpn_model)
    set_for_training(args, tc)
    tc.weight_prediction_network.train()


def setup_optimization(args, tc):
    # TODO deduplicate?
    standard_setup_optimization(args, tc)
    wpn_optimizer_args = args.wpn_optimizer_args
    tc.wpn_optimizer = tc.accelerator.prepare(
        OPTIMIZER_NAME_MAP[args.wpn_optimizer_class](unfrozen_parameters(tc.weight_prediction_network),
                                                     **wpn_optimizer_args))
    wpn_last_batch = tc.last_batch // args.l2w_meta_interval
    if args.wpn_scheduler_class is not None:
        wpn_scheduler_args = deepcopy(args.wpn_scheduler_args)
        if 'T_0' in wpn_scheduler_args:
            wpn_scheduler_args['T_0'] = int(math.ceil(wpn_scheduler_args['T_0'] * wpn_last_batch))
        if 'patience' in wpn_scheduler_args:
            wpn_scheduler_args['patience'] = int(wpn_scheduler_args['patience'] * wpn_last_batch)
        if args.scheduler_class == 'cosine':
            wpn_scheduler_args['T_max'] = wpn_last_batch
        tc.wpn_scheduler = tc.accelerator.prepare(
            SCHEDULER_NAME_MAP[args.wpn_scheduler_class](tc.wpn_optimizer, **wpn_scheduler_args))


def calculate_budget_probs(p, num_heads):
    # magic, unexplained constants are taken from the original implementation:
    # https://github.com/LeapLabTHU/L2W-DEN/blob/06b6f3ea8b08a3af283c14231175a24ee303065f/tools/utils.py#L122
    magic_p = math.log(p / 20)
    probs = torch.exp(magic_p * torch.arange(1, num_heads + 1))
    probs = probs / probs.sum()
    return probs


class MetaSGD(torch.optim.SGD):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        group = self.param_groups[0]
        self.param_ids = {id(p) for p in group['params']}

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']
        for (name, param), grad in zip(self.net.named_parameters(), grads):
            if id(param) not in self.param_ids:
                continue
            if weight_decay != 0:
                grad_wd = grad.add(param.detach(), alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[param]:
                buffer = self.state[param]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1 - dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            set_parameter_by_name(self.net, name, param.detach() - lr * grad_n)


class MetaAdamW(torch.optim.AdamW):
    def __init__(self, net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        group = self.param_groups[0]
        self.param_ids = {id(p) for p in group['params']}

    def meta_step(self, grads):
        group = self.param_groups[0]
        lr = group['lr']
        beta1, beta2 = group['betas']
        weight_decay = group["weight_decay"]
        eps = group['eps']
        maximize = group['maximize']
        amsgrad = group["amsgrad"]
        for (name, param), grad in zip(self.net.named_parameters(), grads):
            if id(param) not in self.param_ids:
                continue
            grad = grad if not maximize else -grad
            state = self.state[param]
            # State initialization
            exp_avg = self.state[param]['exp_avg']
            exp_avg_sq = self.state[param]['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sqs = self.state[param]['max_exp_avg_sq']
            step_t = self.state[param]['step']
            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                exp_avg_sq = torch.view_as_real(exp_avg_sq)
                param = torch.view_as_real(param)
                if amsgrad:
                    max_exp_avg_sqs = torch.view_as_real(max_exp_avg_sqs)
            # update step
            step_t += 1
            # Perform stepweight decay
            param = param * (1 - lr * weight_decay)
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            step = step_t
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()
            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = max_exp_avg_sqs.clone()
                max_exp_avg_sqs.copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            set_parameter_by_name(self.net, name, torch.addcdiv(param.detach(), exp_avg, denom))


def training_loop(args, tc):
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    if args.mixup_alpha is not None or args.cutmix_alpha is not None:
        mixup_mode = 'batch' if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, mode=mixup_mode,
            label_smoothing=mixup_smoothing, num_classes=unwrapped_model.number_of_classes)
    else:
        mixup_fn = None
    head_budget_distribution = calculate_budget_probs(args.l2w_target_p, unwrapped_model.number_of_heads)
    unreduced_criterion = tc.criterion_type(reduction='none')
    criterion = tc.criterion_type(reduction='mean')
    while tc.state.current_batch <= tc.last_batch:
        # save model conditionally
        if tc.accelerator.is_main_process:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        in_training_eval(args, tc)
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(tc.train_loader)
            X, y = next(train_iter)
        if mixup_fn is not None:
            X, y = mixup_fn(X, y)
        # forward
        set_for_training(args, tc)
        # reimplementation, see:
        # https://github.com/LeapLabTHU/L2W-DEN/blob/06b6f3ea8b08a3af283c14231175a24ee303065f/tools/main_imagenet_DDP.py#L325
        if tc.state.current_batch % args.l2w_meta_interval == 0 and tc.state.current_batch > 0:
            # split batch into two parts
            X, X_meta = X.chunk(2)
            y, y_meta = y.chunk(2)
            # create pseudo_net
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            pseudo_net = deepcopy(unwrapped_model)
            pseudo_net.load_state_dict(unwrapped_model.state_dict())
            # pseudo_net = tc.accelerator.prepare(pseudo_net)
            # create a meta-optimizer
            optimizer_type = type(tc.optimizer.optimizer)
            if args.with_backbone:
                pseudo_net.train('all')
            else:
                pseudo_net.train('without_backbone')
            if issubclass(optimizer_type, torch.optim.SGD):
                pseudo_optimizer = MetaSGD(pseudo_net, unfrozen_parameters(pseudo_net), lr=1e-4)
            elif issubclass(optimizer_type, torch.optim.AdamW):
                pseudo_optimizer = MetaAdamW(pseudo_net, unfrozen_parameters(pseudo_net))
            else:
                raise NotImplementedError('Meta step not implemented for other optimizers.')
            pseudo_net.train('all')
            pseudo_optimizer.load_state_dict(tc.optimizer.optimizer.state_dict())
            #
            pseudo_outputs = pseudo_net(X)
            for head_i, head_output in enumerate(pseudo_outputs):
                assert not head_output.isnan().any(), f'{tc.state.current_batch=} {head_i=} {head_output=}'
            pseudo_losses = torch.stack([unreduced_criterion(head_output, y)
                                         for head_output in pseudo_outputs], dim=1)
            # calculate normalized pseudo weights
            pseudo_weights = tc.weight_prediction_network(pseudo_losses.detach())
            pseudo_weights = pseudo_weights - pseudo_weights.mean()
            pseudo_weights = torch.ones_like(pseudo_weights) + args.l2w_epsilon * pseudo_weights
            # calculate weighted loss
            pseudo_weighted_losses = torch.mean(pseudo_weights * pseudo_losses)
            # calculate gradients
            pseudo_grads = torch.autograd.grad(pseudo_weighted_losses, pseudo_net.parameters(),
                                               create_graph=True)
            # and make a (differentiable) meta-step
            pseudo_optimizer.meta_step(pseudo_grads)
            # forward meta batch through the updated pseudo network
            meta_outputs = pseudo_net(X_meta)
            for head_i, head_output in enumerate(meta_outputs):
                assert not head_output.isnan().any(), f'{tc.state.current_batch=} {head_i=} {head_output=}'
            # (meta chunk) samples allocation to heads (see Figure 3 from the paper)
            used_sample_indices = set()
            meta_losses = []
            for head_i in range(unwrapped_model.number_of_heads - 1):
                with torch.no_grad():
                    head_probs = F.softmax(meta_outputs[head_i], dim=1)
                    head_confidence, _ = head_probs.max(dim=1, keepdim=False)
                    _, sorted_indices = head_confidence.sort(dim=0, descending=True)
                    batch_size = sorted_indices.size(0)
                    unused_indices = [x.item() for x in sorted_indices if x.item() not in used_sample_indices]
                    selected_indices = unused_indices[:math.floor(batch_size * head_budget_distribution[head_i])]
                if len(selected_indices) > 0:
                    meta_losses.append(criterion(meta_outputs[head_i][selected_indices],
                                                 y_meta[selected_indices]))
            # last head
            unused_indices = [x.item() for x in sorted_indices if x.item() not in used_sample_indices]
            if len(unused_indices) > 0:
                meta_losses.append(
                    criterion(meta_outputs[unwrapped_model.number_of_heads - 1][unused_indices],
                              y_meta[unused_indices]))
            meta_loss = torch.stack(meta_losses).mean()
            tc.wpn_optimizer.zero_grad(set_to_none=True)
            tc.accelerator.backward(meta_loss)
            if args.clip_grad_norm is not None:
                total_norm = tc.accelerator.clip_grad_norm_(tc.weight_prediction_network.parameters(),
                                                            args.clip_grad_norm)
                if tc.accelerator.is_main_process:
                    tc.writer.add_scalar(f'Train/WPN gradient norm', total_norm.item(),
                                         global_step=tc.state.current_batch)
            tc.wpn_optimizer.step()
            if tc.wpn_scheduler is not None:
                # log LRs
                if tc.accelerator.is_main_process:
                    for i, lr in enumerate(get_lrs(tc.wpn_optimizer)):
                        tc.writer.add_scalar(f'Train/WPN group {i} LR', lr, global_step=tc.state.current_batch)
                if args.wpn_scheduler_class == 'reduce_on_plateau':
                    tc.wpn_scheduler.step(meta_loss)
                else:
                    tc.wpn_scheduler.step()
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Meta loss', meta_loss.item(), global_step=tc.state.current_batch)
            del pseudo_grads
            del unwrapped_model, pseudo_net, pseudo_optimizer, pseudo_outputs
            del pseudo_weights, pseudo_losses, pseudo_weighted_losses
            del X_meta, y_meta, meta_outputs, meta_losses, meta_loss
            del unused_indices, used_sample_indices, head_probs, head_confidence, sorted_indices, selected_indices
            # may fix out of memory problems
            tc.accelerator.free_memory()
        output = tc.model(X)
        for head_i, head_output in enumerate(output):
            assert not head_output.isnan().any(), f'{tc.state.current_batch=} {head_i=} {head_output=}'
        # weighted loss computation
        losses = torch.stack([unreduced_criterion(y_pred, y) for y_pred in output], dim=1)
        with torch.no_grad():
            weights = tc.weight_prediction_network(losses.detach())
            weights = weights - weights.mean()
            weights = torch.ones_like(weights) + args.l2w_epsilon * weights
        loss = torch.mean(weights.detach() * losses)
        # gradient computation and training step
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Train/Loss', loss.item(), global_step=tc.state.current_batch)
        # bookkeeping
        tc.state.current_batch += 1


def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging.')
    tc = L2wTrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()

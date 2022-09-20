import argparse
import json
import logging
import math
import os
from datetime import datetime

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from common import INIT_NAME_MAP, OPTIMIZER_NAME_MAP, LOSS_NAME_MAP, MODEL_NAME_MAP, SCHEDULER_NAME_MAP, mixup, \
    mixup_criterion, get_lrs
from common import parser
from datasets import DATASETS_NAME_MAP
from eval import benchmark_earlyexiting, evaluate_earlyexiting_classification, \
    get_preds_earlyexiting, test_earlyexiting_classification, test_classification
from utils import get_device, get_loader, generate_run_name, load_state, save_state, get_run_id, recreate_base_model


def train_earlyexits_sequentially(_args: argparse.Namespace):
    raise NotImplementedError('TODO')


# TODO deduplicate this code some day
def train_earlyexits(args: argparse.Namespace):
    # learning setup
    base_model, base_args = recreate_base_model(args, args.base_on, args.exp_id)
    model_args = json.loads(args.model_args)
    model = MODEL_NAME_MAP[args.model_class](base_model, **model_args).to(get_device())
    init_fun = INIT_NAME_MAP[args.init_fun if args.init_fun is not None else base_args.init_fun]
    train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset]()
    batch_size = args.batch_size if args.batch_size is not None else base_args.batch_size
    train_loader, train_eval_loader = get_loader(train_data, batch_size), get_loader(train_eval_data, batch_size)
    test_loader = get_loader(test_data, batch_size)
    batches_per_epoch = len(train_loader)
    epochs = args.epochs if args.epochs is not None else base_args.epochs
    last_batch = (epochs) * batches_per_epoch - 1
    eval_points = args.eval_points if args.eval_points is not None else base_args.eval_points
    eval_batch_list = [
        round(x) for x in torch.linspace(0, last_batch, steps=eval_points, device='cpu').tolist()
    ]
    eval_batches = args.eval_batches if args.eval_batches is not None else base_args.eval_batches
    criterion_args = json.loads(args.loss_args)
    criterion_type = LOSS_NAME_MAP[args.loss_type or base_args.loss_type]
    criterion = criterion_type(reduction='mean', **criterion_args)
    optimizer_args = json.loads(args.optimizer_args if args.optimizer_args is not None else base_args.optimizer_args)
    optimizer_class = args.optimizer_class if args.optimizer_class is not None else base_args.optimizer_class
    if args.scheduler_class is not None:
        scheduler_args = json.loads(args.scheduler_args)
        if 'T_0' in scheduler_args:
            scheduler_args['T_0'] = int(math.ceil(scheduler_args['T_0'] * last_batch))
        if 'patience' in scheduler_args:
            scheduler_args['patience'] = int(scheduler_args['patience'] * last_batch)
        if args.scheduler_class == 'cosine':
            scheduler_args['T_max'] = last_batch

    def create_optimizer_scheduler():
        optimizer = OPTIMIZER_NAME_MAP[optimizer_class](model.unfrozen_parameters(), **optimizer_args)
        if args.scheduler_class is not None:
            scheduler = SCHEDULER_NAME_MAP[args.scheduler_class](optimizer, **scheduler_args)
            return optimizer, scheduler
        else:
            return optimizer, None

    # files setup
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    exp_name, run_name = generate_run_name(args)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f'state.pth'
    # logging setup
    if args.use_wandb:
        entity = os.environ['WANDB_ENTITY']
        project = os.environ['WANDB_PROJECT']
        run_id = get_run_id(run_name)
        wandb.tensorboard.patch(root_logdir=str(run_dir.resolve()), tensorboardX=False, pytorch=True, save=False)
        if run_id is not None:
            wandb.init(entity=entity, project=project, id=run_id, resume='must', dir=str(run_dir.resolve()))
        else:
            wandb.init(entity=entity, project=project, config=args, name=run_name,
                       dir=str(run_dir.resolve()))
        wandb.run.log_code('.', include_fn=lambda path: path.endswith('.py'))
    summary_writer = SummaryWriter(str(run_dir.resolve()))
    # state loading
    state = load_state(args, state_path)
    if state is not None:
        model.load_state_dict(state['model_state'])
        if args.with_backbone:
            model.train('all')
        else:
            model.train('only_heads')
        optimizer, scheduler = create_optimizer_scheduler()
        optimizer.load_state_dict(state['optimizer_state'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler_state'])
        current_batch = state['current_batch']
    else:
        state = {}
        state['args'] = args
        if init_fun is not None:
            for head_module in model.head_modules():
                head_module.apply(init_fun)
        current_batch = 0
        model.all_mode()
        if args.with_backbone:
            model.train('all')
        else:
            model.train('only_heads')
        optimizer, scheduler = create_optimizer_scheduler()
    # training
    model_saved = datetime.now()
    train_iter = iter(train_loader)
    while current_batch <= last_batch:
        # save model conditionally
        now = datetime.now()
        if (now - model_saved).total_seconds() > 60 * args.save_every:
            state['current_batch'] = current_batch
            state['model_state'] = model.state_dict()
            state['optimizer_state'] = optimizer.state_dict()
            if scheduler is not None:
                state['scheduler_state'] = scheduler.state_dict()
            save_state(args, state, state_path)
            model_saved = datetime.now()
        # model evaluation
        if current_batch in eval_batch_list:
            summary_writer.add_scalar('Train/Progress', current_batch / last_batch, global_step=current_batch)
            test_losses, test_accs = test_earlyexiting_classification(model,
                                                                      test_loader,
                                                                      criterion_type,
                                                                      batches=eval_batches)
            train_losses, train_accs = test_earlyexiting_classification(model,
                                                                        train_eval_loader,
                                                                        criterion_type,
                                                                        batches=eval_batches)
            for i in range(model.number_of_heads):
                summary_writer.add_scalar(f'Eval/Head {i} test loss', test_losses[i], global_step=current_batch)
                summary_writer.add_scalar(f'Eval/Head {i} test accuracy', test_accs[i], global_step=current_batch)
                summary_writer.add_scalar(f'Eval/Head {i} train loss', train_losses[i], global_step=current_batch)
                summary_writer.add_scalar(f'Eval/Head {i} train accuracy', train_accs[i], global_step=current_batch)
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, y = next(train_iter)
        X = X.to(get_device(), non_blocking=True)
        y = y.to(get_device(), non_blocking=True)
        if args.mixup_alpha > 0.0:
            X, y_a, y_b, lam = mixup(X, y, args.mixup_alpha)
        # training step
        # (without the original head)
        output = model(X)
        for head_output in output:
            assert not head_output.isnan().any(), f'{current_batch=} {head_output=}'
        if args.mixup_alpha > 0.0:
            if args.with_backbone:
                y_head_preds = output
                loss = sum(mixup_criterion(y_pred, y_a, y_b, lam, criterion) for y_pred in
                           y_head_preds) / model.number_of_heads
            else:
                y_head_preds = output[:-1]
                loss = sum(mixup_criterion(y_pred, y_a, y_b, lam, criterion) for y_pred in
                           y_head_preds) / model.number_of_attached_heads
        else:
            if args.with_backbone:
                y_head_preds = output
                loss = sum(criterion(y_pred, y) for y_pred in y_head_preds) / model.number_of_heads
            else:
                y_head_preds = output[:-1]
                loss = sum(criterion(y_pred, y) for y_pred in y_head_preds) / model.number_of_attached_heads
        # TODO refactor into separate functions for readability!
        if args.auxiliary_loss_type == 'distill_last':
            # distill each head to the lass head
            distillation_loss = 0.0
            num_heads = len(output) - 1
            for i in range(num_heads):
                dis_loss = torch.nn.functional.cross_entropy(output[i],
                                                             torch.softmax(output[num_heads].detach(), dim=-1))
                distillation_loss += dis_loss
            distillation_loss /= num_heads
            summary_writer.add_scalar(f'Train/Distillation Loss', distillation_loss.item(), global_step=current_batch)
            loss = loss + args.auxiliary_loss_weight * distillation_loss
        elif args.auxiliary_loss_type == 'distill_later':
            # distill each head to all deeper heads
            distillation_loss = 0.0
            num_outputs = len(output)
            addition_counter = 0
            for i in range(num_outputs):
                for j in range(i + 1, num_outputs):
                    dis_loss = torch.nn.functional.cross_entropy(output[i], torch.softmax(output[j].detach(), dim=-1))
                    distillation_loss += dis_loss
                    addition_counter += 1
            distillation_loss /= addition_counter
            summary_writer.add_scalar(f'Train/Distillation Loss', distillation_loss.item(), global_step=current_batch)
            loss = loss + args.auxiliary_loss_weight * distillation_loss
        elif args.auxiliary_loss_type == 'distill_next':
            # distill each head to the next head
            distillation_loss = 0.0
            num_outputs = len(output)
            for i in range(num_outputs - 1):
                dis_loss = torch.nn.functional.cross_entropy(output[i], torch.softmax(output[i + 1].detach(), dim=-1))
                distillation_loss += dis_loss
            distillation_loss /= num_outputs - 1
            summary_writer.add_scalar(f'Train/Distillation Loss', distillation_loss.item(), global_step=current_batch)
            loss = loss + args.auxiliary_loss_weight * distillation_loss
        elif args.auxiliary_loss_type is not None:
            raise ValueError('Illegal auxiliary_loss_type value')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            # log LRs
            for i, lr in enumerate(get_lrs(optimizer)):
                summary_writer.add_scalar(f'Train/Group {i} LR', lr, global_step=current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                scheduler.step(loss)
            else:
                scheduler.step()
        summary_writer.add_scalar(f'Train/Loss', loss.item(), global_step=current_batch)
        current_batch += 1
    if 'completed' not in state:
        state['current_batch'] = current_batch
        state['model_state'] = model.state_dict()
        state['optimizer_state'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state'] = scheduler.state_dict()
        head_costs, model_params = benchmark_earlyexiting(model, test_loader)
        head_preds, labels = get_preds_earlyexiting(model, test_loader)
        eval_results = evaluate_earlyexiting_classification(head_preds, labels, head_costs, args.eval_thresholds)
        state.update(eval_results)
        state['model_params'] = dict(model_params)
        state['completed'] = True
        save_state(args, state, state_path)


def main():
    args = parser.parse_args()
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging.')
    if args.sequentially:
        train_earlyexits_sequentially(args)
    else:
        train_earlyexits(args)


if __name__ == '__main__':
    main()

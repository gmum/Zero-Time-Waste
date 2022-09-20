import json
import logging
import math
import os
from datetime import datetime

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from common import INIT_NAME_MAP, OPTIMIZER_NAME_MAP, LOSS_NAME_MAP, MODEL_NAME_MAP, SCHEDULER_NAME_MAP, get_lrs, mixup, \
    mixup_criterion
from common import parser
from datasets import DATASETS_NAME_MAP
from eval import test_condconv_classification, benchmark_condconv
from utils import get_device, get_loader, generate_run_name, load_state, save_state, get_run_id


def train(args):
    assert args.k_fractions is not None
    # learning setup
    model_args = json.loads(args.model_args)
    model = MODEL_NAME_MAP[args.model_class](**model_args).to(get_device())
    model.set_fc_k_fractions(args.k_fractions)
    init_fun = INIT_NAME_MAP[args.init_fun]
    train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset]()
    batch_size = args.batch_size
    batches_per_epoch = math.ceil(len(train_data) / batch_size)
    last_batch = args.epochs * batches_per_epoch - 1
    eval_batch_list = [
        round(x) for x in torch.linspace(0, last_batch, steps=args.eval_points, device='cpu').tolist()
    ]
    criterion_args = json.loads(args.loss_args)
    criterion_type = LOSS_NAME_MAP[args.loss_type]
    criterion = criterion_type(reduction='mean', **criterion_args)
    optimizer_args = json.loads(args.optimizer_args)
    optimizer = OPTIMIZER_NAME_MAP[args.optimizer_class](model.parameters(), **optimizer_args)
    if args.scheduler_class is not None:
        scheduler_args = json.loads(args.scheduler_args)
        if 'T_0' in scheduler_args:
            scheduler_args['T_0'] = int(math.ceil(scheduler_args['T_0'] * last_batch))
        if 'patience' in scheduler_args:
            scheduler_args['patience'] = int(scheduler_args['patience'] * last_batch)
        if args.scheduler_class == 'cosine':
            scheduler_args['T_max'] = last_batch
        scheduler = SCHEDULER_NAME_MAP[args.scheduler_class](optimizer, **scheduler_args)
    else:
        scheduler = None
    train_loader, train_eval_loader = get_loader(train_data, batch_size), get_loader(train_eval_data, batch_size)
    test_loader = get_loader(test_data, batch_size)
    # files setup
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    _, run_name = generate_run_name(args)
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
        optimizer.load_state_dict(state['optimizer_state'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler_state'])
        current_batch = state['current_batch']
    else:
        state = {}
        state['args'] = args
        if init_fun is not None:
            model.apply(init_fun)
        current_batch = 0
    # training
    model_saved = datetime.now()
    model.train()
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
            test_losses, test_accs = test_condconv_classification(model,
                                                                  test_loader,
                                                                  criterion_type,
                                                                  k_fractions=args.k_fractions,
                                                                  batches=args.eval_batches)
            train_losses, train_accs = test_condconv_classification(model,
                                                                    train_loader,
                                                                    criterion_type,
                                                                    k_fractions=args.k_fractions,
                                                                    batches=args.eval_batches)
            for k_frac, test_loss, test_acc, train_loss, train_acc in \
                    zip(args.k_fractions, test_losses, test_accs, train_losses, train_accs):
                summary_writer.add_scalar(f'Eval/Test loss ({k_frac=})', test_loss, global_step=current_batch)
                summary_writer.add_scalar(f'Eval/Test accuracy ({k_frac=})', test_acc, global_step=current_batch)
                summary_writer.add_scalar(f'Eval/Train loss ({k_frac=})', train_loss, global_step=current_batch)
                summary_writer.add_scalar(f'Eval/Train accuracy ({k_frac=})', train_acc, global_step=current_batch)
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
        y_pred_dict = model(X)
        loss = 0.0
        for y_pred in y_pred_dict.values():
            if args.mixup_alpha > 0.0:
                loss += mixup_criterion(y_pred, y_a, y_b, lam, criterion)
            else:
                loss += criterion(y_pred, y)
        loss /= len(y_pred_dict)
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
        # test on testset
        test_losses, test_accs = test_condconv_classification(model,
                                                              test_loader,
                                                              criterion_type,
                                                              k_fractions=args.k_fractions)
        for k_frac, test_loss, test_acc in zip(args.k_fractions, test_losses, test_accs):
            summary_writer.add_scalar(f'Eval/Test loss ({k_frac=})', test_loss, global_step=current_batch)
            summary_writer.add_scalar(f'Eval/Test accuracy ({k_frac=})', test_acc, global_step=current_batch)
        state['hyperparam_values'] = args.k_fractions
        state['final_scores'] = test_accs
        state['final_losses'] = test_losses
        # benchmark model efficiency
        k_frac_costs, model_params = benchmark_condconv(model, test_loader, args.k_fractions)
        state['final_flops'] = [cost.total() for cost in k_frac_costs]
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
    train(args)


if __name__ == '__main__':
    main()

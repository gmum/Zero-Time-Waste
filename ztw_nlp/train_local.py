import os
from copy import deepcopy
from pathlib import Path

from common import get_default_args
from datasets_config import DATASET_TO_NUM_CLASSES, DATASET_TO_SEQUENCE_LENGTH, MODEL_TO_TOKENIZER_NAME
from train import train
from utils import generate_run_name
from methods.early_exit import train as train_ee
from dotenv import load_dotenv


# command:
# python -m scripts.local_training_example


def main():
    common_args = get_default_args()

    exp_ids = [1]
    exp_names = []
    display_names = []

    DS = 'dbpedia14'
    MODEL = 'bert_base'
    NUM_ICS = 11

    common_args.use_wandb = False
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.dataset = DS
    common_args.dataset_args = {'tokenizer_name': MODEL_TO_TOKENIZER_NAME[MODEL], 'max_seq_length': DATASET_TO_SEQUENCE_LENGTH[DS]}
    # common_args.dataset_args = {'tokenizer_name': 'bert-base-uncased', 'max_seq_length': 128}

    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0
    common_args.optimizer_class = "adam"
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 2e-5
    common_args.optimizer_args.weight_decay = 0.0005
    common_args.scheduler_class = "cosine"
    common_args.scheduler_args = {}

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = MODEL
    base_model_args.model_args = {'num_classes': DATASET_TO_NUM_CLASSES[DS], 'max_seq_length': DATASET_TO_SEQUENCE_LENGTH[DS]}
    base_model_args.epochs = 5
    base_model_args.batch_size = 16
    base_model_args.eval_points = 9

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        train(args)
        exp_name, run_name = generate_run_name(args)
    exp_names.append(exp_name)
    display_names.append('BERT-base')
    base_exp_name = exp_name

    sdn_model_args = deepcopy(common_args)
    sdn_model_args.epochs = 5
    sdn_model_args.batch_size = 256
    sdn_model_args.eval_points = 9
    sdn_model_args.with_backbone = False
    sdn_model_args.model_class = 'sdn'
    sdn_model_args.model_args = {}
    sdn_model_args.model_args.head_type = 'transformer_standard_head'
    sdn_model_args.model_args.place_at = list(range(NUM_ICS))
    sdn_model_args.optimizer_class = 'adam'
    sdn_model_args.optimizer_args = {}
    sdn_model_args.optimizer_args.lr = 0.005
    sdn_model_args.optimizer_args.weight_decay = 0.0001

    for exp_id in exp_ids:
        args = deepcopy(sdn_model_args)
        args.exp_id = exp_id
        args.base_on = base_exp_name
        train_ee(args)
        exp_name, run_name = generate_run_name(args)
    exp_names.append(exp_name)
    display_names.append(f'SDN')


if __name__ == "__main__":
    load_dotenv('user.env')
    main()

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from datasets import load_dataset
from src.args import Config
from src.consts import TASK_TO_SENTENCE_KEY, TASK_TO_VAL_SPLIT_NAME


class TokenizedDataset(Dataset):
    def __init__(self, bert_path: str, task: str, split: str):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path, use_fast=True)
        dataset = load_dataset("glue", task, split=split)
        sentence1_key, sentence2_key = TASK_TO_SENTENCE_KEY[task]

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding="max_length", max_length=128, truncation=True)

            return result

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc=f"Running tokenizer on {task}:{split}",
        )
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    train_set = TokenizedDataset(bert_path=config.model_path, task=config.task, split="train")
    val_set = TokenizedDataset(
        bert_path=config.model_path,
        task=config.task,
        split=TASK_TO_VAL_SPLIT_NAME[config.task],
    )

    train_data_loader = DataLoader(
        train_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_data_loader = DataLoader(
        val_set,
        batch_size=config.val_batch_size,
        num_workers=config.num_workers,
    )

    return train_data_loader, val_data_loader


def get_validation_data_loaders_for_ee(config: Config) -> DataLoader:
    val_set = TokenizedDataset(
        bert_path=config.model_path,
        task=config.task,
        split=TASK_TO_VAL_SPLIT_NAME[config.task],
    )
    if config.limit_val_batches is not None:
        val_set = Subset(val_set, range(config.limit_val_batches))

    val_data_loader = DataLoader(val_set, batch_size=1, num_workers=config.num_workers)

    return val_data_loader

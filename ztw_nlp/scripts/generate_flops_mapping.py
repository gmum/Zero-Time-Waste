import json
import os
from pathlib import Path

import click
import torch
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from scripts.flops.hacked_modeling_bert import (
    BertForSequenceClassification as HackedBertForSequenceClassification,
)
from src.consts import TASK_TO_SENTENCE_KEY
from src.models.ee.simple import SimpleEarlyExitModel
from src.models.ee.ztw import ZTWModel


@click.command("Generates json file mapping index of exit layer to flops used for single "
               "forward pass for base BERT model, PABEE, SDN and ZTW.")
@click.option(
    "-o",
    "--output_file",
    type=Path,
    default="results/flops_mapping.json",
    help="Path to save json file with computed flops.",
)
def main(output_file: Path):
    num_labels = 2
    config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    bert = HackedBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", config=config, from_tf=False
    )

    tasks = ["rte", "mrpc", "qnli", "qqp", "sst2"]
    outputs = {}
    for task in tqdm(tasks):
        dataset = load_dataset("glue", task, split="validation")
        sentence1_key, sentence2_key = TASK_TO_SENTENCE_KEY[task]

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)

            return result

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc=f"Running tokenizer on {task}:validation",
        )
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        data_loader = DataLoader(
            dataset,
            shuffle=True,
            num_workers=12,
        )

        task_flops = {}
        with torch.no_grad():
            for batch in data_loader:
                inputs = (batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

                task_flops["base"] = {}
                flops = (
                    FlopCountAnalysis(bert, inputs)
                    .uncalled_modules_warnings(False)
                    .unsupported_ops_warnings(False)
                    .total()
                )
                task_flops["base"]["total"] = flops

                for i in range(config.num_hidden_layers):
                    bert.bert.encoder.exit_layer = i
                    flops = (
                        FlopCountAnalysis(bert.bert, inputs)
                        .uncalled_modules_warnings(False)
                        .unsupported_ops_warnings(False)
                        .total()
                    )
                    task_flops["base"][i + 1] = flops
                bert.bert.encoder.exit_layer = None

                model_outputs = bert(*inputs, output_hidden_states=True)
                states = torch.stack(model_outputs.hidden_states[1:], dim=1)

                task_flops["simple"] = {}
                simple_ee = SimpleEarlyExitModel(
                    num_hidden_layers=config.num_hidden_layers,
                    hidden_size=config.hidden_size,
                    num_labels=config.num_labels,
                )

                for exit_layer_idx in range(config.num_hidden_layers):
                    simple_ee.exit_layer = exit_layer_idx
                    flops = (
                        FlopCountAnalysis(simple_ee, states)
                        .uncalled_modules_warnings(False)
                        .unsupported_ops_warnings(False)
                        .total()
                    )
                    task_flops["simple"][exit_layer_idx + 1] = (
                        flops + task_flops["base"][exit_layer_idx + 1]
                    )

                task_flops["ztw"] = {}
                ztw = ZTWModel(
                    num_hidden_layers=config.num_hidden_layers,
                    hidden_size=config.hidden_size,
                    num_labels=config.num_labels,
                )

                for exit_layer_idx in range(config.num_hidden_layers):
                    ztw.exit_layer = exit_layer_idx
                    flops = (
                        FlopCountAnalysis(ztw, states)
                        .uncalled_modules_warnings(False)
                        .unsupported_ops_warnings(False)
                        .total()
                    )
                    task_flops["ztw"][exit_layer_idx + 1] = (
                        flops + task_flops["base"][exit_layer_idx + 1]
                    )

                task_flops["base"] = {"total": task_flops["base"]["total"]}
                outputs[task] = task_flops
                break

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open("w+") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    main()

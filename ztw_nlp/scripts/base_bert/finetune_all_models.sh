#!/bin/bash

# This script trains base BERT on GLUE tasks

export CUDA_VISIBLE_DEVICES=0

./scripts/base_bert/finetune_single_model.sh RTE 50 16 1 2e-5 5 0;
./scripts/base_bert/finetune_single_model.sh MRPC 50 16 1 2e-5 5 0;
./scripts/base_bert/finetune_single_model.sh SST2 200 16 1 1e-5 3 0;
./scripts/base_bert/finetune_single_model.sh QQP 2000 32 4 5e-5 3 0;
./scripts/base_bert/finetune_single_model.sh QNLI 500 16 1 1e-5 3 0;

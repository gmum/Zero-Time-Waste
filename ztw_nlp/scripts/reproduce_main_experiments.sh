#!/bin/bash

#./scripts/base_bert/finetune_all_models.sh

for seed in 0 1 2; do
  ./scripts/ee/train_single_task.sh RTE ${seed} 1e-3 1e-4;
  ./scripts/ee/train_single_task.sh MRPC ${seed} 1e-3 1e-4;
  ./scripts/ee/train_single_task.sh SST2 ${seed} 1e-3 1e-4;
  ./scripts/ee/train_single_task.sh QNLI ${seed} 1e-3 1e-4;
  ./scripts/ee/train_single_task.sh QQP ${seed} 1e-3 1e-4;
done

#!/bin/bash

task=$1
seed=$2
lr=$3
epochs=$4
patience=$5
batch_size=$6

pabee_dir="models/pabee_${task}_${seed}_${lr}"
python src/train.py \
  --ee_model "pabee" \
  --early_exit_criterion patience \
  --tags "${task}_final" \
  --task $task \
  --model_path "models/${task}_seed_0" \
  --output_dir $pabee_dir \
  --seed $seed \
  --lr $lr \
  --max_epochs $epochs \
  --patience $patience \
  --train_batch_size $batch_size \
  --num_workers 20 \
  --evaluate_ee_thresholds;
rm -rf $pabee_dir
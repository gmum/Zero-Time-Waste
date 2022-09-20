#!/bin/bash

task=$1
seed=$2
lr=$3
epochs=$4
patience=$5
batch_size=$6

ztw_dir="models/ztw_${task}_${seed}_${lr}"
python src/train.py \
  --ee_model "ztw" \
  --tags "${task}_final" \
  --task $task \
  --model_path "models/${task}_seed_0" \
  --output_dir $ztw_dir \
  --seed $seed \
  --lr $lr \
  --max_epochs $epochs \
  --patience $patience \
  --train_batch_size $batch_size \
  --num_workers 20 \
  --evaluate_ee_thresholds;
rm -rf $ztw_dir
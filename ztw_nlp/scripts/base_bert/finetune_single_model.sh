#!/bin/bash

# Finetunes model on given task with provided set of hyperparameters

task_name=$1
logging_steps=$2
batch_size=$3
gradient_accumulation_steps=$4
learning_rate=$5
num_train_epochs=$6
seed=$7
output_dir="models/${task_name}_seed_${seed}"

mkdir -p ${output_dir}

#bert-base-uncased
python src/run_glue.py \
  --model_name_or_path "bert-base-uncased" \
  --task_name "${task_name}" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size "${batch_size}" \
  --per_device_eval_batch_size 1 \
  --learning_rate "${learning_rate}" \
  --save_strategy "no" \
  --save_steps "${logging_steps}" \
  --evaluation_strategy "no" \
  --eval_steps "${logging_steps}" \
  --logging_strategy "steps" \
  --logging_steps 1 \
  --num_train_epochs "${num_train_epochs}" \
  --output_dir "${output_dir}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --seed "${seed}"


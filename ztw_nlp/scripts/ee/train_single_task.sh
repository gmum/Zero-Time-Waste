#!/bin/bash

export PYTHONPATH="."

task=$1
seed=$2
lr=$3
ens_lr=$4
EPOCHS=50
PATIENCE=5
BATCH_SIZE=256

./scripts/ee/train_pabee.sh ${task} ${seed} ${lr} ${EPOCHS} ${PATIENCE} ${BATCH_SIZE}
./scripts/ee/train_sdn.sh ${task} ${seed} ${lr} ${EPOCHS} ${PATIENCE} ${BATCH_SIZE}
./scripts/ee/train_ztw_ens.sh ${task} ${seed} ${lr} ${EPOCHS} ${PATIENCE} ${BATCH_SIZE} ${ens_lr}

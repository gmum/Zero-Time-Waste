#!/bin/bash
#SBATCH --job-name=ztw
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-gpu=10
source user.env

#for XPID in 1 2 3 4 5; do
for XPID in 1; do

  #  MODEL="resnet18"

  #  MODEL="resnet34"

  #  MODEL="resnet50"
  #  MODEL_ARGS='{"num_classes":10}'
  #  MODEL_ARGS='{"num_classes":100}'
#  INIT_FUN=""

  MODEL="wideresnet"
  MODEL_ARGS='{"num_blocks":[5,5,5],"widen_factor":4,"num_classes":100,"dropout_rate":0.0,"stride_on_beginning":false}'
  INIT_FUN="--init_fun wideresnet_init"

#  MODEL="efficientnetv2"
#  MODEL_ARGS='{"model_name":"s","n_classes":100}'
#  INIT_FUN=""

  #  DATASET="cifar10"
  DATASET="cifar100"
  BATCH_SIZE=128
  LOSS_TYPE="ce"
  LOSS_ARGS='{"label_smoothing":0.0}'
  OPTIMIZER="sgd"

  SCHEDULER="cosine"
  SCHEDULER_ARGS='{}'
  MIXUP_ALPHA=0.0
  EPOCHS=100

#  for lr in 0.1 0.01; do
  for lr in 0.1; do
#    for wd in 0.05 0.005 0.0005 0.0005 0.0; do
    for wd in 0.0005; do
      OPTIMIZER_ARGS='{"lr":'$lr',"momentum":0.9,"weight_decay":'$wd'}'
      # train base network
      ARGS="--exp_id $XPID --use_wandb \
            --model_class $MODEL --model_args $MODEL_ARGS $INIT_FUN \
            --dataset $DATASET --batch_size $BATCH_SIZE --epochs $EPOCHS \
            --loss_type $LOSS_TYPE --loss_args $LOSS_ARGS \
            --optimizer_class $OPTIMIZER --optimizer_args $OPTIMIZER_ARGS \
            --scheduler_class $SCHEDULER --scheduler_args $SCHEDULER_ARGS \
            --mixup_alpha $MIXUP_ALPHA"
      srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m train $ARGS &
    done
  done
done
wait

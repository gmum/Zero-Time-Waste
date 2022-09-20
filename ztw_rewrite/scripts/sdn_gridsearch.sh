#!/bin/bash
#SBATCH --job-name=ztw
#SBATCH --qos=big
#SBATCH --gres=gpu:3
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-gpu=10
source user.env

#for XPID in 1 2 3 4 5; do
for XPID in 1; do

  DATASET="cifar100"

  BATCH_SIZE=128
  LOSS_TYPE="ce"
  LOSS_ARGS='{"label_smoothing":0.0}'
  OPTIMIZER="sgd"
  OPTIMIZER_ARGS='{"lr":0.1,"momentum":0.9,"weight_decay":0.0005}'
  SCHEDULER="cosine"
  SCHEDULER_ARGS='{}'
  MIXUP_ALPHA=0.0
  EPOCHS=150

#  MODEL="resnet50"
#  MODEL_ARGS='{"num_classes":100}'
  MODEL="wideresnet"
  MODEL_ARGS='{"num_blocks":[5,5,5],"widen_factor":4,"num_classes":100,"dropout_rate":0.0,"stride_on_beginning":false}'
  INIT_FUN="--init_fun wideresnet_init"

  ARGS="--exp_id $XPID --use_wandb \
        --model_class $MODEL --model_args $MODEL_ARGS $INIT_FUN \
        --dataset $DATASET --batch_size $BATCH_SIZE --epochs $EPOCHS \
        --loss_type $LOSS_TYPE --loss_args $LOSS_ARGS \
        --optimizer_class $OPTIMIZER --optimizer_args $OPTIMIZER_ARGS \
        --scheduler_class $SCHEDULER --scheduler_args $SCHEDULER_ARGS \
        --mixup_alpha $MIXUP_ALPHA"
  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m train $ARGS &
  wait
  BASE_ON=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)

  # heads-training specific settings
  PLACE_AT="[2,4,6,8,10,12,14]"
  HEAD_EPOCHS=100
  SDN_MODEL_ARGS="{\"head_type\":\"conv\",\"place_at\":$PLACE_AT}"
  TRAIN_BACKBONE=""
  for lr in 0.1 0.01 0.001; do
#  for lr in 0.001; do
    for wd in 0.005 0.0005 0.00005 0.0; do
#    for wd in 0.0; do
      HEAD_OPTIMIZER_ARGS='{"lr":'$lr',"momentum":0.9,"weight_decay":'$wd'}'
      # train base network
      MODEL="sdn"
      ARGS="--exp_id $XPID --use_wandb \
            --model_class $MODEL --model_args $SDN_MODEL_ARGS \
            --dataset $DATASET --batch_size $BATCH_SIZE --epochs $HEAD_EPOCHS \
            --loss_type $LOSS_TYPE --loss_args $LOSS_ARGS \
            --optimizer_class $OPTIMIZER --optimizer_args $HEAD_OPTIMIZER_ARGS \
            --scheduler_class $SCHEDULER --scheduler_args $SCHEDULER_ARGS \
            --mixup_alpha $MIXUP_ALPHA \
            --base_on $BASE_ON $TRAIN_BACKBONE"
      srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m methods.early_exit $ARGS &
      WAIT_FOR+=$!" "
      RUN_NAMES+=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)" "
    done
  done
done
wait

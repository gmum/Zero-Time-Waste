#!/bin/bash
#SBATCH --job-name=ztw
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=10
source user.env

function run_single_seed {
  XPID=$1

  DATASET="tinyimagenet"

  BATCH_SIZE=128
  LOSS_TYPE="ce"
  LOSS_ARGS='{"label_smoothing":0.0}'
  OPTIMIZER="sgd"
  OPTIMIZER_ARGS='{"lr":0.1,"momentum":0.9,"weight_decay":0.0005}'
  SCHEDULER_TYPE="cosine"
  SCHEDULER_ARGS="{}"
  SCHEDULER="--scheduler_class $SCHEDULER_TYPE --scheduler_args $SCHEDULER_ARGS"
  MIXUP_ALPHA=0.0
  EPOCHS=150

#  MODEL="wideresnet"
#  MODEL_ARGS='{"num_blocks":[5,5,5],"widen_factor":4,"num_classes":200,"dropout_rate":0.3,"stride_on_beginning":false}'
#  INIT_FUN="--init_fun wideresnet_init"
#  MODEL="resnet101"
  MODEL="resnet50"
  MODEL_ARGS='{"num_classes":200}'
#  MODEL="vgg16bn"
#  MODEL_ARGS='{"input_size":64,"num_classes":200}'
#  INIT_FUN="--init_fun vgg_init"

  # heads-training specific settings
#  PLACE_AT="[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]"
  PLACE_AT="[1,2,3,4,5,6,7,8,9,10,11,12,13,14]"
#  PLACE_AT="[1,3,6,9,12]"
  HEAD_EPOCHS=100
  HEAD_OPTIMIZER_ARGS='{"lr":0.1,"momentum":0.9,"weight_decay":0.00005}'
  SDN_MODEL_ARGS="{\"head_type\":\"conv\",\"place_at\":$PLACE_AT}"
  CASCADING_MODEL_ARGS="{\"head_type\":\"conv_cascading\",\"place_at\":$PLACE_AT}"

  # ensemble-training specific settings
  RENS_BATCH_SIZE=256
  RENS_EPOCHS=100
  RENS_OPTIMIZER="adam"
  RENS_SCHEDULER="--scheduler_class $SCHEDULER_TYPE --scheduler_args $SCHEDULER_ARGS"

  RUN_NAMES=""
  WAIT_FOR=""

  # train base network
  ARGS="--exp_id $XPID --use_wandb \
        --model_class $MODEL --model_args $MODEL_ARGS $INIT_FUN \
        --dataset $DATASET --batch_size $BATCH_SIZE --epochs $EPOCHS \
        --loss_type $LOSS_TYPE --loss_args $LOSS_ARGS \
        --optimizer_class $OPTIMIZER --optimizer_args $OPTIMIZER_ARGS \
        $SCHEDULER \
        --mixup_alpha $MIXUP_ALPHA"
  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m train $ARGS &
  wait
  BASE_ON=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)
  RUN_NAMES+=$BASE_ON" "

  for TRAIN_BACKBONE in ""; do
    MODEL="sdn"
    ARGS="--exp_id $XPID --use_wandb \
          --model_class $MODEL --model_args $SDN_MODEL_ARGS \
          --dataset $DATASET --batch_size $BATCH_SIZE --epochs $HEAD_EPOCHS \
          --loss_type $LOSS_TYPE --loss_args $LOSS_ARGS \
          --optimizer_class $OPTIMIZER --optimizer_args $HEAD_OPTIMIZER_ARGS \
          $SCHEDULER \
          --mixup_alpha $MIXUP_ALPHA \
          --base_on $BASE_ON $TRAIN_BACKBONE $AUXILIARY_LOSS_TYPE $AUXILIARY_WEIGHT"
    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m methods.early_exit $ARGS &
#    RUN_NAMES+=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)" "

    # train ZTW cascading
    MODEL="ztw_cascading"
    ARGS="--exp_id $XPID --use_wandb \
          --model_class $MODEL --model_args $CASCADING_MODEL_ARGS \
          --dataset $DATASET --batch_size $BATCH_SIZE --epochs $HEAD_EPOCHS \
          --loss_type $LOSS_TYPE --loss_args $LOSS_ARGS \
          --optimizer_class $OPTIMIZER --optimizer_args $HEAD_OPTIMIZER_ARGS \
          $SCHEDULER \
          --mixup_alpha $MIXUP_ALPHA \
          --base_on $BASE_ON $TRAIN_BACKBONE $AUXILIARY_LOSS_TYPE $AUXILIARY_WEIGHT"
    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m methods.early_exit $ARGS &
    WAIT_FOR+=$!" "
    BASE_ON_GRID=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)
    RUN_NAMES+=$BASE_ON_GRID" "

    wait $WAIT_FOR

    # train arithmetic ensembling
    MODEL="ztw_ensembling"
    RENS_MODEL_ARGS="{\"type\":\"arithmetic\"}"
    for lr in 0.0005; do
      RENS_OPTIMIZER_ARGS="{\"lr\":$lr}"
      ARGS="--exp_id $XPID --use_wandb \
          --model_class $MODEL --model_args $RENS_MODEL_ARGS \
          --dataset $DATASET --batch_size $RENS_BATCH_SIZE --epochs $RENS_EPOCHS \
          --loss_type nll --loss_args {} \
          --optimizer_class $RENS_OPTIMIZER --optimizer_args $RENS_OPTIMIZER_ARGS \
          $RENS_SCHEDULER \
          --mixup_alpha $MIXUP_ALPHA \
          --base_on $BASE_ON_GRID $TRAIN_BACKBONE "
      srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m methods.early_exit $ARGS &
      RUN_NAMES+=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)" "
    done

    # train geometric ensembling
    MODEL="ztw_ensembling"
    RENS_MODEL_ARGS="{\"type\":\"geometric\"}"
    for lr in 0.0005; do
#    for lr in 0.5 0.05 0.005 0.0005 0.00005; do
      RENS_OPTIMIZER_ARGS="{\"lr\":$lr}"
      ARGS="--exp_id $XPID --use_wandb \
          --model_class $MODEL --model_args $RENS_MODEL_ARGS \
          --dataset $DATASET --batch_size $RENS_BATCH_SIZE --epochs $RENS_EPOCHS \
          --loss_type ce --loss_args {} \
          --optimizer_class $RENS_OPTIMIZER --optimizer_args $RENS_OPTIMIZER_ARGS \
          $RENS_SCHEDULER \
          --mixup_alpha $MIXUP_ALPHA \
          --base_on $BASE_ON_GRID $TRAIN_BACKBONE "
      srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python -m methods.early_exit $ARGS &
      RUN_NAMES+=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)" "
    done
  done

  wait
  echo $RUN_NAMES
}

#XPIDS="1 2 3 4 5"
XPIDS="1 3 5"
#XPIDS="1"

IFS=', ' read -r -a XPIDS_ARRAY <<< "$XPIDS"
for XPID in "${XPIDS_ARRAY[@]:1}"; do
  run_single_seed "$XPID" &
done
EXP_NAMES=$(run_single_seed ${XPIDS_ARRAY[0]})
echo EXP_NAMES: $EXP_NAMES

# performance plots for those same runs
OUTPUT_DIR="figures_ensembling_resnet50_tinyimagenet"
#DISPLAY_NAMES="Base Cascading Arithmetic Geometric"
#DISPLAY_NAMES="Base Cascading Arithmetic Geometric_0.5 Geometric_0.05 Geometric_0.005 Geometric_0.0005 Geometric_0.00005"
DISPLAY_NAMES="Base Cascading Arithmetic Geometric"
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_plot --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_plot --mode calibration --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_plot --mode ood_detection --ood_dataset cifar10 --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb

# tables
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_table --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb

# HI plots
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.hindsight_improvability --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb
#!/bin/bash
#SBATCH --job-name=ztw
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=10
source user.env

function run_single_seed {
  XPID=$1

#  DATASET="cifar100"
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
#  MODEL_ARGS='{"num_blocks":[5,5,5],"widen_factor":4,"num_classes":100,"dropout_rate":0.3,"stride_on_beginning":false}'
#  MODEL_ARGS='{"num_blocks":[5,5,5],"widen_factor":4,"num_classes":200,"dropout_rate":0.3,"stride_on_beginning":false}'
#  INIT_FUN="--init_fun wideresnet_init"
  MODEL="resnet50"
#  MODEL_ARGS='{"num_classes":100}'
  MODEL_ARGS='{"num_classes":200}'

  # heads-training specific settings
#  PLACE_AT="[2,4,6,8,10,12,14]"
  PLACE_AT="[1,2,3,4,5,6,7,8,9,10,11,12,13,14]"
  HEAD_EPOCHS=100
  HEAD_OPTIMIZER_ARGS='{"lr":0.1,"momentum":0.9,"weight_decay":0.00005}'
  SDN_MODEL_ARGS="{\"head_type\":\"conv\",\"place_at\":$PLACE_AT}"
  CASCADING_MODEL_ARGS="{\"head_type\":\"conv_cascading\",\"place_at\":$PLACE_AT}"

  # ensemble-training specific settings
  RENS_BATCH_SIZE=256
  RENS_EPOCHS=100
  RENS_MODEL_ARGS="{}"
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

#  for TRAIN_BACKBONE in "" "--with_backbone"; do
  for TRAIN_BACKBONE in ""; do
    for AUXILIARY_LOSS_TYPE in "" "--auxiliary_loss_type distill_last" \
      "--auxiliary_loss_type distill_later" \ "--auxiliary_loss_type distill_next"; do
      for AUXILIARY_WEIGHT in "--auxiliary_loss_weight 1.0"; do
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
        RUN_NAMES+=$(singularity exec $SINGULARITY_ARGS $SIF_PATH python -m get_run_name $ARGS)" "

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
        # train ensembling
        MODEL="ztw_ensembling"
        for lr in 0.0005; do
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
    done
  done

  wait
  echo $RUN_NAMES
}

#XPIDS="1 2 3 4 5"
XPIDS="1 2 3"
#XPIDS="1"

IFS=', ' read -r -a XPIDS_ARRAY <<< "$XPIDS"
for XPID in "${XPIDS_ARRAY[@]:1}"; do
  run_single_seed "$XPID" &
done
EXP_NAMES=$(run_single_seed ${XPIDS_ARRAY[0]})
echo EXP_NAMES: $EXP_NAMES

# performance plots for those same runs
OUTPUT_DIR="figures_distillation_resnet50_tinyimagenet"
DISPLAY_NAMES="Base SDN Cascading ZTW SDN_last Cascading_last ZTW_last SDN_later Cascading_later ZTW_later SDN_next Cascading_next ZTW_next"

# cost vs plots
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_plot --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_plot --mode calibration --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_plot --mode ood_detection --ood_dataset cifar10 --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb

# tables
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.cost_vs_table --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb

# HI plots
singularity exec $SINGULARITY_ARGS $SIF_PATH python -m visualize.hindsight_improvability --exp_names $EXP_NAMES --exp_ids $XPIDS --display_names $DISPLAY_NAMES --output_dir $OUTPUT_DIR --use_wandb
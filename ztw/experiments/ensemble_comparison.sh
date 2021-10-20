#!/bin/bash

#SBATCH --job-name=ens_comp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task 10

declare -A num_heads

num_heads=(["resnet56"]=27 ["vgg16bn"]=14 ["wideresnet32_4"]=15 ["mobilenet"]=13 ["tv_resnet"]=5)

seed=1700
arch="resnet56"
dataset="tinyimagenet"
heads="${num_heads[$arch]}"
echo "arch: $arch dataset: $dataset heads: $heads seed: $seed"

python train_networks.py -d $dataset -a $arch -t cnn -s $seed --heads all --tag "${seed}_${dataset}_${arch}_base_cnn" --skip_train_logits --save_test_logits
python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv sdn_pool --heads all --stacking --detach_prev --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_heads" &
wait

# geometric
ens_type=geometric
for head_id in $(seq 0 $heads); do
  python train_networks.py -d $dataset -a $arch -t running_ensb -s $seed --head_arch conv sdn_pool --save_test_logits --skip_train_logits --stacking --detach_prev --head_ids $head_id --run_ensb_type "${ens_type}" --alpha 0. --tag "${seed}_${dataset}_${arch}_running_ensb_geo" --parent_id "child_of_${seed}_${dataset}_${arch}_heads" --heads all &
  if ! (((head_id + 1) % 2)); then
    wait
  fi
done
# additive
ens_type=additive
for head_id in $(seq 0 $heads); do
  python train_networks.py -d $dataset -a $arch -t running_ensb -s $seed --head_arch conv sdn_pool --save_test_logits --skip_train_logits --stacking --detach_prev --head_ids $head_id --run_ensb_type "${ens_type}" --alpha 0. --tag "${seed}_${dataset}_${arch}_running_ensb_arith" --parent_id "child_of_${seed}_${dataset}_${arch}_heads" --heads all &
  if ! (((head_id + 1) % 2)); then
    wait
  fi
done
#unweighted
ens_type=standard
for head_id in $(seq 0 $heads); do
  python train_networks.py -d $dataset -a $arch -t running_ensb -s $seed --head_arch conv sdn_pool --save_test_logits --skip_train_logits --stacking --detach_prev --head_ids $head_id --run_ensb_type "${ens_type}" --run_ensb_epochs 1 --alpha 0. --tag "${seed}_${dataset}_${arch}_running_ensb_std" --parent_id "child_of_${seed}_${dataset}_${arch}_heads" --heads all &
  if ! (((head_id + 1) % 2)); then
    wait
  fi
done

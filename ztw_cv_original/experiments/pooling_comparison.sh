#!/bin/bash

#SBATCH --job-name=pool_comp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task 10

declare -A num_heads

num_heads=(["resnet56"]=27 ["vgg16bn"]=14 ["wideresnet32_4"]=15 ["mobilenet"]=13 ["tv_resnet"]=5)

seed=1600
arch="resnet56"
dataset="tinyimagenet"
heads="${num_heads[$arch]}"
echo "arch: $arch dataset: $dataset heads: $heads seed: $seed"

size_after_pool=4
# avg pooling
pooling=avg_pool
python train_networks.py -d $dataset -a $arch -t cnn -s $seed --heads all --tag "${seed}_${dataset}_${arch}_base_cnn" --skip_train_logits --save_test_logits
python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv $pooling --size_after_pool $size_after_pool --stacking --detach_prev --heads all --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_${pooling}_heads" &
wait
for head_id in $(seq 0 $heads); do
  python train_networks.py -d $dataset -a $arch -t running_ensb -s $seed --head_arch conv $pooling --size_after_pool $size_after_pool --save_test_logits --skip_train_logits --stacking --detach_prev --head_ids $head_id --alpha 0. --tag "${seed}_${dataset}_${arch}_running_ensb" --parent_id "child_of_${seed}_${dataset}_${arch}_${pooling}_heads" --heads all &
  if ! (((head_id + 1) % 2)); then
    wait
  fi
done
# max pooling
pooling=max_pool
python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv $pooling --size_after_pool $size_after_pool --stacking --detach_prev --heads all --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_${pooling}_heads" &
wait
for head_id in $(seq 0 $heads); do
  python train_networks.py -d $dataset -a $arch -t running_ensb -s $seed --head_arch conv $pooling --size_after_pool $size_after_pool --save_test_logits --skip_train_logits --stacking --detach_prev --head_ids $head_id --alpha 0. --tag "${seed}_${dataset}_${arch}_running_ensb" --parent_id "child_of_${seed}_${dataset}_${arch}_${pooling}_heads" --heads all &
  if ! (((head_id + 1) % 2)); then
    wait
  fi
done

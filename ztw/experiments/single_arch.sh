#!/bin/bash

declare -A num_heads

num_heads=( ["resnet56"]=27 ["vgg16bn"]=14 ["wideresnet32_4"]=15 ["mobilenet"]=13 ["tv_resnet"]=5 )

if [[ -n $3 ]]; then
    seed=$3
else
    seed=1666
fi
echo "seed: $seed"

if [ -z "$1" ] | [ -z "$2" ]; then
    echo "No arch or dataset passed!"
    exit 1
fi

arch=$1
dataset=$2
heads="${num_heads[$arch]}"
echo "arch: $arch dataset: $dataset heads: $heads"

if [[ $arch == "tv_resnet" ]] && [[ $dataset != "imagenet" ]]; then
    echo "tv_resnet can be trained only on imagenet!"
    exit 1
fi

# if [[ $arch == "tv_resnet" ]]; then
#     python train_networks.py -d $dataset -a $arch -t cnn -s $seed --heads third --head_arch conv_less_ch sdn_pool --skip_train_logits -p 4 --tag "${seed}_${dataset}_${arch}_base_cnn"
#     python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --heads third --head_arch conv_less_ch sdn_pool --skip_train_logits --suffix 40ep_pooling4_lr2_third_ln_shift1 -p 4 --lr_scaler 2 --head_shift 1 --tag "${seed}_${dataset}_${arch}_sdn_baseline"
#     python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --heads third --head_arch conv_less_ch sdn_pool --skip_train_logits --stacking --detach_prev --detach_norm layernorm --suffix 40ep_pooling4_lr2_third_ln_shift1 -p 4 --lr_scaler 2 --head_shift 1 --tag "${seed}_${dataset}_${arch}_heads"
# else
#     python train_networks.py -d $dataset -a $arch -t cnn -s $seed --heads all --tag "${seed}_${dataset}_${arch}_base_cnn" --skip_train_logits --save_test_logits
#     python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv sdn_pool --heads all --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_sdn_baseline" &
#     python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv sdn_pool --heads all --stacking --detach_prev --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_heads" &
#     wait
# fi


for head_id in $(seq 0 $heads); do
    python train_networks.py -d $dataset -a $arch -t running_ensb -s $seed --head_arch conv sdn_pool --save_test_logits --skip_train_logits --stacking --detach_prev --head_ids $head_id --alpha 0. --tag "${seed}_${dataset}_${arch}_running_ensb" --parent_id "child_of_${seed}_${dataset}_${arch}_heads" --heads all &
    if ! (( (head_id + 1) % 2)); then
        wait
    fi
done
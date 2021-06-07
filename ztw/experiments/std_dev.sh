#!/bin/bash

#SBATCH --job-name=std_devs
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task 10

##SBATCH --job-name=std_devs
##SBATCH --qos=big
##SBATCH --gres=gpu:2
##SBATCH --mem=120G
##SBATCH --cpus-per-task 20

seed_displcmt=200
num_seeds=10
max_iter=$(($num_seeds/2))

arch=resnet56
dataset=tinyimagenet
seed=210
python train_networks.py -d $dataset -a $arch -t cnn -s $seed --heads all --tag "${seed}_${dataset}_${arch}_base_cnn" --skip_train_logits --save_test_logits
python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv sdn_pool --heads all --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_sdn_baseline" &
python train_networks.py -d $dataset -a $arch -t sdn_ic -s $seed --head_arch conv sdn_pool --heads all --stacking --detach_prev --save_test_logits --skip_train_logits --tag "${seed}_${dataset}_${arch}_heads" &
wait

for arch in resnet56 vgg16bn wideresnet32_4 mobilenet; do
    # for dataset in cifar10 cifar100 tinyimagenet; do
    for dataset in cifar10 cifar100; do
        for i in $(seq 0 $max_iter); do
            seed1=$(($i+$seed_displcmt))
            seed2=$(($max_iter+$i+$seed_displcmt))
            CUDA_VISIBLE_DEVICES=0 bash ./experiments/single_arch.sh $arch $dataset $seed1 &
            CUDA_VISIBLE_DEVICES=1 bash ./experiments/single_arch.sh $arch $dataset $seed2 &
            wait
        done
    done
done

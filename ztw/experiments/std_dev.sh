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
max_iter=$(($num_seeds / 2))

for arch in resnet56 vgg16bn wideresnet32_4 mobilenet; do
  for dataset in cifar10 cifar100 tinyimagenet; do
    for i in $(seq 0 $max_iter); do
      seed1=$(($i + $seed_displcmt))
      seed2=$(($max_iter + $i + $seed_displcmt))
      CUDA_VISIBLE_DEVICES=0 bash ./experiments/single_arch.sh $arch $dataset $seed1 &
      CUDA_VISIBLE_DEVICES=1 bash ./experiments/single_arch.sh $arch $dataset $seed2 &
      wait
    done
  done
done

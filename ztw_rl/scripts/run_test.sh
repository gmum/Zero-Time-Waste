#!/bin/bash

#SBATCH --job-name=template
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=10

# change according to your setup
source $HOME/miniconda3/bin/activate std_pt
cd $HOME/ztw_rl
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/test.py
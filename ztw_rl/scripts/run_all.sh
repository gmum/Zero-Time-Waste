#!/bin/bash

#SBATCH --job-name=template
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=10

# change according to your setup
source $HOME/miniconda3/bin/activate std_pt
cd $HOME/ztw_rl
# ==============================
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py PongNoFrameskip-v4 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py MsPacman-v0 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py Breakout-v0 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py Qbert-v0 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py Phoenix-v0 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py Riverraid-v0 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py Seaquest-v0 &
CUDA_VISIBLE_DEVICES=0 NEPTUNE_PROJECT='user/ztw-rl' python scripts/small_ic.py AirRaid-v0 &
wait
# ==============================

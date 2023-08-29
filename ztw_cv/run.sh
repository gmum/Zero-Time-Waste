#!/bin/bash
source user.env
eval "$(conda shell.bash hook)"
conda activate ee_eval_env
python -m scripts.$1
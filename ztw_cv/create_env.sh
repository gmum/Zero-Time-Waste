#!/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n ee_eval_env python=3.10
  conda activate ee_eval_env
  conda install gcc=11.3.0 gxx=11.3.0 ninja -c conda-forge
  conda install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti -c nvidia/label/cuda-11.7.1
  conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia/label/cuda-11.7.1
  mkdir pip-build
  TMPDIR=pip-build pip --no-input --no-cache-dir install accelerate scikit-learn fvcore
  TMPDIR=pip-build pip --no-input --no-cache-dir install submitit omegaconf
  TMPDIR=pip-build pip --no-input --no-cache-dir install wandb tensorboard seaborn
  rm -rf pip-build
  conda env export | grep -v "^prefix: " > environment.yml
fi

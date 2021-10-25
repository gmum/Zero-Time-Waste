### Source Code for "Zero Time Waste: Recycling Predictions in Early Exit Neural Networks"

Paper link: [https://arxiv.org/pdf/2106.05409.pdf](https://arxiv.org/pdf/2106.05409.pdf)

The repository is split into two parts: supervised learning experiments (directory `ztw`) and reinforcement learning
experiments (directory `ztw_rl`). The supervised learning code is based on
the [code](https://github.com/yigitcankaya/Shallow-Deep-Networks) from
the [Shallow-Deep-Networks paper](https://arxiv.org/abs/1810.07052).

To run the experiments:

1. Preferably, install a conda environment with Python 3.8: `conda create -n ztw_env python=3.8 swig=4.0.2`.
2. Activate the environment with `conda activate ztw_env`.
3. Install the dependencies from the `requirements.txt` file into your python environment
   with `pip install -r requirements.txt`.
4. Set up a [neptune.ai](https://neptune.ai/) account and create a project.
5. Set the `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` environment variables.
6. Optionally fetch the TinyImageNet dataset. (Note, however, that the original link to that dataset is not valid
   anymore.)
7. To run the supervised learning experiments, execute the `ztw/experiments/std_dev.sh`.
8. To run the reinforcement learning experiments, execute the `ztw_rl/scripts/run_all.sh` script.
9. To run the ablation experiments, edit and execute the `ztw/experiments/ensemble_comparison.sh`
   and `ztw/experiments/pooling_comparison.sh` scripts.
10. To generate the plots use the notebooks from the `ztw/notebooks` and `ztw_rl/notebooks` directories.

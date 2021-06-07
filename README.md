### Source Code for "Zero Time Waste: Recycling Predictions in Early Exit Neural Networks"

Paper link: TODO

The repository is split into two parts: supervised learning experiments (directory `ztw`) and reinforcement learning experiments (directory `ztw_rl`). The supervised learning code is based on the [code](https://github.com/yigitcankaya/Shallow-Deep-Networks) from the [Shallow-Deep-Networks paper](https://arxiv.org/abs/1810.07052).

To run the experiments:
1. Install the dependencies from the `requirements.txt` file into your python/conda environment.
2. Set up a [neptune.ai](https://neptune.ai/) account, create a project and change the project name in the code where necessary.
3. Optionally fetch the TinyImageNet dataset. (Note however that the original link to that dataset is not valid anymore.)
4. To run the supervised learning experiments, execute the `ztw/experiments/std_dev.sh`.
5. To run the reinforcement learning experiments, execute the `ztw_rl/scripts/run_all.sh` script.
6. To run the ablation experiments, edit and execute the `ztw/experiments/ensemble_comparison.sh` and `ztw/experiments/pooling_comparison.sh` scripts.
7. To generate the plots use the notebooks from the `ztw/notebooks` and `ztw_rl/notebooks` directories.




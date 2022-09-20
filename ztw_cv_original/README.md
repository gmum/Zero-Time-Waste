### Source Code for computer vision experiments for "Zero Time Waste: Recycling Predictions in Early Exit Neural Networks"

The code is based on
the [code](https://github.com/yigitcankaya/Shallow-Deep-Networks) from
the [Shallow-Deep-Networks paper](https://arxiv.org/abs/1810.07052).

Note that the `requirements.txt` and the created environment are shared with `ztw_rl` part of the experiments. 

To run the experiments:
1. Preferably, install a conda environment with Python 3.8: `conda create -n ztw_env python=3.8 swig=4.0.2`.
2. Activate the environment with `conda activate ztw_env`.
3. Install the dependencies from the `requirements.txt` file into your python environment
   with `pip install -r requirements.txt`.
4. Set up a [neptune.ai](https://neptune.ai/) account and create a project.
5. Set the `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` environment variables.
6. Optionally fetch the TinyImageNet dataset. (Note, however, that the original link to that dataset is not valid
   anymore.)
7. To run the computer vision experiments execute the `experiments/std_dev.sh` script.
8. To generate the plots use the notebook from the `notebooks` directory.

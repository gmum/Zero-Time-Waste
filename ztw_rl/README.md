### Source Code for reinforcement learning experiments for "Zero Time Waste: Recycling Predictions in Early Exit Neural Networks"

Note that the `requirements.txt` and the created environment are shared with `ztw_cv_original` part of the experiments.

To run the experiments:
1. Preferably, install a conda environment with Python 3.8: `conda create -n ztw_env python=3.8 swig=4.0.2`.
2. Activate the environment with `conda activate ztw_env`.
3. Install the dependencies from the `requirements.txt` file into your python environment
   with `pip install -r requirements.txt`.
4. Set up a [neptune.ai](https://neptune.ai/) account and create a project.
5. Set the `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` environment variables.
6. To run the reinforcement learning experiments execute the `scripts/run_all.sh` script.
9. To generate the plots use the notebook from the `notebooks` directory.

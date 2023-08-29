# Description
TODO
  
# Running
0. Make sure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) is installed on your server, and that it's present in `$PATH`. You may want to install it by yourself, or ask whether conda is already installed somewhere and just add it to `$PATH`.
0. If `ee_eval_env` is not visible in conda, e.g it's a fresh conda installation, create this environment:
`bash create_env.sh`
1. Optionally create a [W&B](https://wandb.ai/) account and add the following content to your `~/.bashrc`:
`export WANDB_API_KEY="<YOUR_KEY>"`
2. Copy the user_example.env file and fill in the paths:
`cp user_example.env user.env`
`vi user.env`
3. Edit the submitit/slurm run script to run the experiments you need:
`vi scripts/your_run_script_name.py`
4. Run the experiment using that script with slurm:
`bash run.sh your_run_script_name`

# Notes
A few things to keep in mind:
- The code generates a unique run name based on the command line arguments passed to the script. When adding new CLI argument that should not affect the run name you have to update the `generate_run_name()` function appropriately.
- The weights are saved every N minutes.
- The training will continue from the last checkpoint if the run with the generated name is present.
- Use the `use_wandb` flag to log and save models to W&B.
- Remember that changing the code will not change the generated experiment name.

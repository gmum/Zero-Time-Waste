# Description
This is a rewrite of Zero Time Waste (ZTW) early exiting method. The code is shorter, clearer
and should be more friendly for new users. Running one of the scripts from the `scripts` directory will
run all the training stages and generate the resulting plots and tables automatically.
  
# Running
0. Build the singularity image if needed:
`TMPDIR=~/tmp/singularity_tmpdir singularity build --fakeroot image.sif image.def`
(You may need to follow [this](https://sylabs.io/guides/3.5/admin-guide/user_namespace.html#fakeroot-feature) to be able to do this locally without root)
1. Optionally create a [W&B](https://wandb.ai/) account and add the following content to your `~/.bashrc`:
`export WANDB_API_KEY="<YOUR_KEY>"`
2. Copy the user_example.env file and fill in the paths:
`cp user_example.env user.env`
`vi user.env`
3. Edit the slurm run script to run the experiments you need:
`vi scripts/run.sh`
4. Run the experiment using that script with slurm:
`sbatch -p some_partition scripts/run.sh`
or directly:
`bash scripts/run.sh`

# Notes
A few things to keep in mind:
- The code generates a unique run name based on the command line arguments passed to the script. When adding new CLI argument remember to update the `generate_run_name()` function appropriately.
- The weights are saved every N minutes.
- The training will continue from the last checkpoint if the run with the generated name is present.
- Use the `--use_wandb` flag to log and save/load models to W&B.
- Change the `XPID` argument to generate a different run name if needed. Remember that changing the code will not change the generated experiment name.
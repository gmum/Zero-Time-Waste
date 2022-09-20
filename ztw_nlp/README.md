# Source Code for natural language processing experiments for "Zero Time Waste: Recycling Predictions in Early Exit Neural Networks"

Extension of Zero Time Waste model to NLP.

## Setup

1. Create and activate conda environment:

```shell
conda env create -f environment.yml
conda activate ztw_nlp
```

2. Set up a [wandb.ai](https://wandb.ai/) account and create a project. 
3. Add your API keys by running `wandb login` as described [here](https://docs.wandb.ai/quickstart).
4. Create `.env` file containing wandb entity and project names, following the example below:
```
WANDB_ENTITY=<entity>
WANDB_PROJECT=<project_name>
```

## Experiments

### Finetune base BERT models on GLUE tasks

```shell
./scripts/base_bert/finetune_all_models.sh
```

### Train ZTW, SDN and PABEE models on single task

```shell
./scripts/ee/train_single_task.sh <task_name> <seed> <main_lr> <ensemble_lr>
```

### Reproduce experiments from paper

```shell
./scripts/reproduce_main_experiments.sh
```

### Process the results from wandb to the format in paper

1. Generate file with flops mapping:

```shell
PYTHONPATH=$PYTHONPATH:. python scripts/generate_flops_mapping.py
```

2. Generate csv file with results for given task and latex code from wandb runs - example for a
   RTE model:

```shell
PYTHONPATH=$PYTHONPATH:. python scripts/generate_final_table.py \
  --task RTE \
  --wandb_tag RTE_final \
  --flops_mapping_path results/flops_mapping.json \
  --output_path results/rte.csv
```
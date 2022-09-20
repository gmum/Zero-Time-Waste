import json
import tempfile
from functools import partial
from math import floor
from pathlib import Path
from typing import Dict

import click
import pandas as pd
import wandb
from tqdm import tqdm
import dotenv
import os

dotenv.load_dotenv()


def _parse_run(run: wandb.apis.public.Run, flops_mapping: Dict) -> Dict:
    name = run.config["ee_model"]
    output_row = {"Model": name}

    table = [a for a in run.logged_artifacts() if "avg_layer_vs_accuracy_table" in a.name][0]
    with tempfile.TemporaryDirectory() as tmpdir:
        table.download(tmpdir)
        table_json = list(Path(tmpdir).joinpath("validation").glob("*.json"))[0]
        with table_json.open() as f:
            table_data = json.load(f)
    df = pd.DataFrame(columns=table_data["columns"], data=table_data["data"])
    df["Average_layer"] *= 12

    bert_flops = flops_mapping["base"]["total"]
    model_flops = flops_mapping["ztw"] if name == "ztw" else flops_mapping["simple"]
    df["Bert flops"] = bert_flops
    _map_to_flops = partial(map_to_flops, flops_mapping=model_flops)
    df["Average flops"] = df["Average_layer"].apply(_map_to_flops)
    df["Budget"] = df["Average flops"] / df["Bert flops"]

    thresholds = [0.25, 0.5, 0.75, 1.0]
    budget, acc = df["Budget"].to_list(), df["Accuracy"].to_list()
    for thresh in thresholds:
        below_thresh = [pair for pair in zip(budget, acc) if pair[0] < thresh]
        try:
            best_val = max(below_thresh, key=lambda pair: pair[1])
        except:
            best_val = [None, None]  # PABEE in some cases might not go below 25% threshold
        output_row[str(100 * thresh)] = best_val[1]

    output_row["Max"] = acc[-1]

    integrated_accuracy = 0.0
    layer_vs_accuracy = list(zip(budget, acc))
    for p1, p2 in zip(layer_vs_accuracy, layer_vs_accuracy[1:]):
        avg_layer_step = p2[0] - p1[0]
        avg_accuracy_height = (p1[1] + p2[1]) / 2
        integrated_accuracy += avg_accuracy_height * avg_layer_step

    integrated_accuracy /= budget[-1] - budget[0]
    output_row["Mean"] = integrated_accuracy

    return output_row


def map_to_flops(avg_layer: float, flops_mapping: Dict):
    base = floor(avg_layer)
    out = flops_mapping[str(base)]
    if base != 12:
        next_diff = flops_mapping[str(base + 1)] - flops_mapping[str(base)]
        remainder = avg_layer - base
        out += remainder * next_diff
    return out


@click.command()
@click.option("-t", "--task", type=str, required=True, help="Task name.")
@click.option("-w", "--wandb_tag", type=str, required=True,
              help="Wandb tag to find runs to parse.")
@click.option("-f", "--flops_mapping_path", type=Path, default="results/flops_mapping.json",
              help="Path to flops mapping file. "
                   "Can be generated with `scripts/generate_flops_mapping.py`.")
@click.option("-o", "--output_path", type=Path, required=True,
              help="Path to save output csv file.")
def main(task: str, wandb_tag: str, flops_mapping_path: Path, output_path: Path):
    api = wandb.Api()
    entity, project = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]
    runs = api.runs(entity + "/" + project)
    runs = [r for r in runs if wandb_tag in r.tags]

    with flops_mapping_path.open() as f:
        flops_dict = json.load(f)[task.lower()]

    results = [_parse_run(run, flops_dict) for run in tqdm(runs)]
    df = pd.DataFrame(results)
    keys = df.columns[1:]
    df = (
        df.groupby(by=["Model"])
            .agg({k: ["mean", "std"] for k in keys})
            .multiply(100)
            .reset_index()
            .round(2)
            .sort_values(by=["Model"])
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False)

    print("Latex parsed results:")
    model_names = df["Model"].unique()
    model_name_to_abbr = {"pabee": "PBEE", "sdn": "SDN", "ztw": "ZTW"}
    for model_name in model_names:
        output_str = f"& {model_name_to_abbr[model_name]}"
        for key in keys:
            thresh_result = df[(df["Model"] == model_name)]
            mean = thresh_result[(key, "mean")].values[0]
            std = thresh_result[(key, "std")].values[0]
            output_str += f" & ${round(mean, 1)}\\pm{round(std, 1)}$"
        output_str += " \\\\"
        print(output_str)


if __name__ == "__main__":
    main()

# %%
import argparse
import json
import subprocess
import sys
from itertools import combinations
from math import prod
from pathlib import Path

import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.data_pipeline.dataset_configs import DATASET_CONFIGS
from src.training.run_experiments2 import MODELS
from src.training.train2 import DEFAULT_TRAINING_CONFIGS, infer_monitor_mode

# %%

# grid of hyperparameter values to search over for each model — ParameterGrid generates every combination
PARAM_GRIDS = {
    "lr":  {"learning_rate": [1e-4, 1e-3, 1e-2]},                                                                   # 3 combos
    "mlp": {"learning_rate": [1e-4, 1e-3, 1e-2], "hidden_dim": [64, 128, 256], "dropout": [0.0, 0.1, 0.3]},        # 27 combos
    "cpd": {"learning_rate": [1e-4, 1e-3, 1e-2], "rank": [4, 8, 16, 32]},                                          # 12 combos
    "tt":  {"learning_rate": [1e-4, 1e-3, 1e-2], "rank": [4, 8, 16, 32]},                                          # 12 combos
    "tr":  {"learning_rate": [1e-4, 1e-3, 1e-2], "rank": [4, 8, 16, 32]},                                          # 12 combos
    "mba": {"learning_rate": [1e-4, 1e-3, 1e-2]},  # interaction_order added dynamically per dataset based on parameter budget
    "rf":  {"max_depth": [None, 5, 10, 20], "max_features": ["sqrt", "log2", 0.5], "min_samples_leaf": [1, 4, 8]},  # 36 combos
}

# %%

# prints the full command before running it so you can see exactly what train2 is being called with, then raises an error immediately if the subprocess fails
def run_command(command):
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

# %%

# counts the total number of explicit parameters in an MBA model up to a given interaction order
def count_mba_parameters(feature_dims, num_classes, interaction_order):
    total = num_classes
    for order in range(1, interaction_order + 1):
        for interaction in combinations(feature_dims, order):
            total += num_classes * prod(interaction)
    return total

# returns the MBA interaction orders that fit under the parameter budget for a given dataset
def allowed_mba_orders(metadata_path, max_parameters, max_order):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_dims = metadata["cardinalities"]
    num_classes = len(metadata["target_mapping"])
    max_order = min(max_order, len(feature_dims))
    return [
        order
        for order in range(1, max_order + 1)
        if count_mba_parameters(feature_dims, num_classes, order) <= max_parameters
    ]

# %%

# reads the best validation score for one grid search combo from its result folder
def read_trial_score(result_dir, metric, mode):
    metadata_path = result_dir / "run_metadata.json"
    if metadata_path.exists(): # prefer the best_model_score saved by the checkpoint callback during training
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        score = metadata.get("best_model_score")
        if score is not None:
            return float(score)

    metrics = pd.read_csv(result_dir / "metrics.csv") # fall back to reading metrics.csv if no checkpoint score exists (e.g. rf)
    values = metrics[metric].dropna()
    if values.empty:
        raise ValueError(f"No values for {metric} in {result_dir}")

    if mode == "min": # return the best value depending on whether we are minimizing or maximizing
        return float(values.min())
    return float(values.max())

# %% 

def parse_args():
    parser = argparse.ArgumentParser(description="Grid search hyperparameter tuning on the tuning split.")

    parser.add_argument("--dataset", dest="datasets", action="append", choices=sorted(DATASET_CONFIGS.keys()), help="Dataset to tune on. Repeat for multiple.")
    parser.add_argument("--model", dest="models", action="append", choices=MODELS, help="Model to tune. Repeat for multiple.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per combo. Defaults to model default.")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-mba-parameters", type=int, default=5_000_000, help="Maximum MBA parameters allowed — filters which interaction orders are included in the grid.")
    parser.add_argument("--max-mba-order", type=int, default=2, help="Maximum MBA interaction order considered.")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument("--metric", type=str, default="val_loss", help="Validation metric to optimise across combos.")
    parser.add_argument("--metric-mode", choices=["min", "max"], default=None, help="Whether the metric should be minimized or maximized. Inferred from metric name if not set.")
    parser.add_argument("--processed-root", type=Path, default=Path("src/data_pipeline/data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/summary_results/results/tuning"), help="Where to save best_params JSON files.")
    parser.add_argument("--skip-preprocessing", action="store_true", default=False, help="Skip make_dataset2 calls and assume processed data already exists.")

    return parser.parse_args()

# %% loops over all model/dataset combinations, runs every hyperparameter combo on the tuning split and saves the best params to a JSON file

def main():
    args = parse_args()

    selected_datasets = args.datasets or sorted(DATASET_CONFIGS.keys()) # defaults to all datasets if none specified
    selected_models = args.models or MODELS # defaults to all models if none specified

    for mi, model in enumerate(selected_models):
        print(f"\n{'='*60}\nMODEL {mi+1}/{len(selected_models)}: {model.upper()}\n{'='*60}", flush=True)

        effective_metric = "val_balanced_acc" if model == "rf" and args.metric == "val_loss" else args.metric # rf has no loss so fall back to balanced acc
        effective_mode = args.metric_mode or infer_monitor_mode(effective_metric) # infer min/max from metric name if not set
        selected_epochs = None if model == "rf" else args.epochs or DEFAULT_TRAINING_CONFIGS[model]["epochs"]

        for di, dataset in enumerate(selected_datasets):
            print(f"\n  [{model.upper()}] dataset {di+1}/{len(selected_datasets)}: {dataset}", flush=True)

            if not args.skip_preprocessing: # runs make_dataset2 to generate tuning and fold splits for this dataset
                run_command([
                    sys.executable, "-m", "src.data_pipeline.make_dataset2",
                    "--dataset", dataset,
                    "--representation", "both",
                    "--output-dir", str(args.processed_root),
                ])

            grid = dict(PARAM_GRIDS[model]) # copy so we don't modify the original
            if model == "mba": # compute which interaction orders fit the parameter budget for this dataset and add them to the grid
                metadata_path = args.processed_root / f"{dataset}_tensor_metadata.json"
                orders = allowed_mba_orders(metadata_path, args.max_mba_parameters, args.max_mba_order)
                if not orders:
                    raise ValueError(f"No MBA interaction order fits --max-mba-parameters={args.max_mba_parameters} for {dataset}.")
                print(f"Allowed MBA interaction orders for {dataset}: {orders}", flush=True)
                grid["interaction_order"] = orders

            combos = list(ParameterGrid(grid)) # generate all combinations
            print(f"Running {len(combos)} combos for {model.upper()} on {dataset}", flush=True)

            best_score = None
            best_params = None

            for ci, params in enumerate(combos):
                result_version = f"tuning_{dataset}_{model}_combo{ci}" # unique folder name for each combo
                print(f"\n  combo {ci+1}/{len(combos)}: {params}", flush=True)

                if model == "rf":
                    command = [
                        sys.executable, "-m", "src.training.train2",
                        "--dataset", dataset,
                        "--model", "rf",
                        "--processed-dir", str(args.processed_root),
                        "--seed", str(args.seed),
                        "--num-workers", str(args.num_workers),
                        "--mode", "tuning", # use the 20% tuning split, not a CV fold
                        "--result-version", result_version,
                        "--skip-test", # no need to test during tuning — we only care about val score
                        "--max-features", str(params["max_features"]),
                        "--min-samples-leaf", str(params["min_samples_leaf"]),
                    ]
                    if params["max_depth"] is not None:
                        command.extend(["--max-depth", str(params["max_depth"])])
                else:
                    command = [
                        sys.executable, "-m", "src.training.train2",
                        "--dataset", dataset,
                        "--model", model,
                        "--processed-dir", str(args.processed_root),
                        "--seed", str(args.seed),
                        "--result-version", result_version,
                        "--epochs", str(selected_epochs),
                        "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                        "--accelerator", args.accelerator,
                        "--learning-rate", str(params["learning_rate"]),
                        "--patience", str(args.patience),
                        "--monitor", effective_metric,
                        "--monitor-mode", effective_mode,
                        "--mode", "tuning", # use the 20% tuning split, not a CV fold
                        "--skip-test", # no need to test during tuning — we only care about val score
                    ]
                    if "rank" in params:
                        command.extend(["--rank", str(params["rank"])])
                    if "interaction_order" in params:
                        command.extend(["--interaction-order", str(params["interaction_order"])])
                    if "hidden_dim" in params:
                        command.extend(["--hidden-dim", str(params["hidden_dim"])])
                    if "dropout" in params:
                        command.extend(["--dropout", str(params["dropout"])])

                run_command(command)

                result_dir = Path("src/summary_results/results") / model / result_version
                score = read_trial_score(result_dir, effective_metric, effective_mode)

                if best_score is None or (effective_mode == "min" and score < best_score) or (effective_mode == "max" and score > best_score):
                    best_score = score # update best score and params when this combo beats the previous best
                    best_params = params

            # save best params to JSON so run_experiments2 can load them for the CV phase
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / f"{model}_{dataset}_best_params.json"
            output = {
                "dataset": dataset,
                "model": model,
                "seed": args.seed,
                "metric": effective_metric,
                "metric_mode": effective_mode,
                "best_score": best_score,
                "best_params": best_params,
                "num_combos": len(combos),
                "epochs": selected_epochs,
                "patience": args.patience if model != "rf" else None,
                "batch_size": args.batch_size if model != "rf" else None,
            }
            output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {output_path}")
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

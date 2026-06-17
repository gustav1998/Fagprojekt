# %%

import json
import subprocess
import sys
import argparse

from pathlib import Path
from src.data_pipeline.dataset_configs import DATASET_CONFIGS
from src.training.train2 import DEFAULT_TRAINING_CONFIGS

# %%

MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tr", "rf"]
N_FOLDS = 5

# %%

# run one subprocess command and stop on failure
def run_command(command):
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

# %%

def parse_args():
    parser = argparse.ArgumentParser(description="Run full CV pipeline for multiple datasets and models.")

    parser.add_argument("--dataset", dest="datasets", action="append", choices=sorted(DATASET_CONFIGS.keys()), help="Dataset to run. Repeat for multiple.")
    parser.add_argument("--model", dest="models", action="append", choices=MODELS, help="Model to run. Repeat for multiple.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument("--processed-root", type=Path, default=Path("src/data_pipeline/data/processed"))
    parser.add_argument("--tuning-dir", type=Path, default=Path("src/summary_results/results/tuning"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--interaction-order", type=int, default=None)
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=True)
    parser.add_argument("--disable-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--monitor-mode", choices=["min", "max"], default=None)
    parser.add_argument("--skip-preprocessing", action="store_true", default=False, help="Skip make_dataset calls and assume processed data already exists.")

    return parser.parse_args()

# %% This runs the full CV pipeline for multiple datasets and models
# It first preprocesses the data for each dataset, then trains each model across all 5 folds

def main():
    args = parse_args()

    selected_datasets = args.datasets or sorted(DATASET_CONFIGS.keys()) # defaults to all 34 datasets if none specified
    selected_models = args.models or MODELS # defaults to all 7 models if none specified
    processed_dir = args.processed_root

    for di, dataset in enumerate(selected_datasets):
        if not args.skip_preprocessing: # runs make_dataset2 to generate the tuning and fold splits for this dataset
            run_command([
                sys.executable, "-m", "src.data_pipeline.make_dataset2",
                "--dataset", dataset,
                "--representation", "both",
                "--output-dir", str(processed_dir),
            ])

        for mi, model in enumerate(selected_models):
            for fold in range(1, N_FOLDS + 1): # loops over each of the 5 folds — each fold is one independent train/test split
                print(
                    f"\n=== dataset {di+1}/{len(selected_datasets)}: {dataset}"
                    f" | model {mi+1}/{len(selected_models)}: {model.upper()}"
                    f" | fold {fold}/{N_FOLDS} ===",
                    flush=True,
                )

                if model == "rf":
                    command = [ # builds the train2 command for RF with the current fold number
                        sys.executable, "-m", "src.training.train2",
                        "--dataset", dataset,
                        "--model", "rf",
                        "--processed-dir", str(processed_dir),
                        "--seed", str(args.seed),
                        "--num-workers", str(args.num_workers),
                        "--mode", "fold", # tells train2 to load fold splits rather than tuning splits
                        "--fold", str(fold), # specifies which fold (1-5) to load
                    ]
                    tuning_json = args.tuning_dir / f"rf_{dataset}_best_params.json"
                    if tuning_json.exists(): # loads best hyperparameters from tuning phase if available
                        rf_params = json.loads(tuning_json.read_text(encoding="utf-8"))["best_params"]
                        command.extend(["--max-features", str(rf_params.get("max_features", "sqrt"))])
                        command.extend(["--min-samples-leaf", str(rf_params.get("min_samples_leaf", 1))])
                        if rf_params.get("max_depth") is not None:
                            command.extend(["--max-depth", str(rf_params["max_depth"])])
                    run_command(command)
                    continue

                config = DEFAULT_TRAINING_CONFIGS[model]
                tuning_json = args.tuning_dir / f"{model}_{dataset}_best_params.json"
                tuned = {}
                if tuning_json.exists(): # loads best hyperparameters from tuning phase if available
                    tuned = json.loads(tuning_json.read_text(encoding="utf-8")).get("best_params", {})

                command = [ # builds the train2 command for neural models with the current fold number
                    sys.executable, "-m", "src.training.train2",
                    "--dataset", dataset,
                    "--model", model,
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--accelerator", args.accelerator,
                    "--processed-dir", str(processed_dir),
                    "--seed", str(args.seed),
                    "--mode", "fold", # tells train2 to load fold splits rather than tuning splits
                    "--fold", str(fold), # specifies which fold (1-5) to load
                    "--epochs", str(args.epochs or config["epochs"]),
                    "--learning-rate", str(tuned.get("learning_rate") or args.learning_rate or config["learning_rate"]),
                    "--patience", str(args.patience),
                    "--monitor", args.monitor,
                ]

                if not args.early_stopping:
                    command.append("--disable-early-stopping")
                if args.monitor_mode is not None:
                    command.extend(["--monitor-mode", args.monitor_mode])
                if model in {"cpd", "tt", "tr"}: # rank is only used by tensor decomposition models
                    command.extend(["--rank", str(tuned.get("rank") or args.rank or config["rank"])])
                if model == "mba": # interaction_order is only used by mba
                    command.extend(["--interaction-order", str(tuned.get("interaction_order") or args.interaction_order or config["interaction_order"])])
                if model == "mlp" and tuned.get("hidden_dim") is not None: # only pass mlp-specific args if tuning found values for them
                    command.extend(["--hidden-dim", str(tuned["hidden_dim"])])
                if model == "mlp" and tuned.get("dropout") is not None:
                    command.extend(["--dropout", str(tuned["dropout"])])

                run_command(command)


if __name__ == "__main__":
    main()

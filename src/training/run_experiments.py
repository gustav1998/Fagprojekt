from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from src.data_pipeline.dataset_configs import DATASET_CONFIGS
from src.training.train import DEFAULT_TRAINING_CONFIGS


MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tr", "rf"]

# run one subprocess command and stop on failure
def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


@click.command()
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    type=click.Choice(sorted(DATASET_CONFIGS.keys())),
    help="Dataset to run. Repeat the option for multiple datasets.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    type=click.Choice(MODELS),
    help="Model to run. Repeat the option for multiple models.",
)
@click.option(
    "--seed",
    "seeds",
    multiple=True,
    type=int,
    default=(42,),
    show_default=True,
    help="Seed to run. Repeat the option for multiple seeds.",
)
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cpu", "gpu"]),
    default="cpu",
    show_default=True,
)
@click.option(
    "--processed-root",
    type=click.Path(path_type=Path),
    default=Path("src/data_pipeline/data/processed"),
    show_default=True,
)
@click.option(
    "--tuning-dir",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/tuning"),
    show_default=True,
)
@click.option("--epochs", type=int, default=None)
@click.option("--learning-rate", type=float, default=None)
@click.option("--rank", type=int, default=None)
@click.option("--interaction-order", type=int, default=None)
@click.option(
    "--early-stopping/--disable-early-stopping",
    default=True,
    show_default=True,
)
@click.option("--patience", type=int, default=15, show_default=True)
@click.option("--monitor", type=str, default="val_loss", show_default=True)
@click.option(
    "--monitor-mode",
    type=click.Choice(["min", "max"]),
    default=None,
)

# Note: This script runs the full pipeline for multiple seeds, datasets, and models. It first preprocesses the data for each seed and dataset, then trains each model on the processed data. 
def main(
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    batch_size: int,
    num_workers: int,
    accelerator: str,
    processed_root: Path,
    tuning_dir: Path,
    epochs: int | None,
    learning_rate: float | None,
    rank: int | None,
    interaction_order: int | None,
    early_stopping: bool,
    patience: int,
    monitor: str,
    monitor_mode: str | None,
) -> None:
    
    # Run preprocessing and training for multiple seeds
    selected_datasets = datasets or tuple(sorted(DATASET_CONFIGS))
    selected_models = models or tuple(MODELS)

    # For each seed, preprocess the data for each dataset, then train each model on the processed data. The processed data is stored in a separate directory for each seed to avoid conflicts.
    for seed in seeds:
        processed_dir = processed_root / f"seed_{seed}"

        for dataset in selected_datasets:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.data_pipeline.make_dataset",
                    "--dataset",
                    dataset,
                    "--representation",
                    "both",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(processed_dir),
                ]
            )

            # After preprocessing, train each selected model on the processed data for the current seed and dataset. The training command includes all relevant hyperparameters, with defaults taken from DEFAULT_TRAINING_CONFIGS if not specified.
            for model in selected_models:
                if model == "rf":
                    command = [
                        sys.executable,
                        "-m",
                        "src.training.train",
                        "--dataset",
                        dataset,
                        "--model",
                        "rf",
                        "--processed-dir",
                        str(processed_dir),
                        "--seed",
                        str(seed),
                        "--num-workers",
                        str(num_workers),
                        "--result-version",
                        f"{dataset}_seed{seed}",
                    ]
                    tuning_json = tuning_dir / f"rf_{dataset}_seed42_best_params.json"
                    if tuning_json.exists():
                        rf_params = json.loads(tuning_json.read_text(encoding="utf-8"))["best_params"]
                        command.extend(["--max-features", str(rf_params.get("max_features", "sqrt"))])
                        command.extend(["--min-samples-leaf", str(rf_params.get("min_samples_leaf", 1))])
                        if rf_params.get("max_depth") is not None:
                            command.extend(["--max-depth", str(rf_params["max_depth"])])
                    run_command(command)
                    continue

                config = DEFAULT_TRAINING_CONFIGS[model]
                tuning_json = tuning_dir / f"{model}_{dataset}_seed42_best_params.json"
                tuned = {}
                if tuning_json.exists():
                    tuned = json.loads(tuning_json.read_text(encoding="utf-8")).get("best_params", {})
                command = [
                    sys.executable,
                    "-m",
                    "src.training.train",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--batch-size",
                    str(batch_size),
                    "--num-workers",
                    str(num_workers),
                    "--accelerator",
                    accelerator,
                    "--processed-dir",
                    str(processed_dir),
                    "--seed",
                    str(seed),
                    "--result-version",
                    f"{dataset}_seed{seed}",
                    "--epochs",
                    str(epochs or config["epochs"]),
                    "--learning-rate",
                    str(tuned.get("learning_rate") or learning_rate or config["learning_rate"]),
                    "--patience",
                    str(patience),
                    "--monitor",
                    monitor,
                ]

                if not early_stopping:
                    command.append("--disable-early-stopping")
                if monitor_mode is not None:
                    command.extend(["--monitor-mode", monitor_mode])
                if model in {"cpd", "tt", "tr"}:
                    command.extend(["--rank", str(tuned.get("rank") or rank or config["rank"])])
                if model == "mba":
                    command.extend(
                        [
                            "--interaction-order",
                            str(tuned.get("interaction_order") or interaction_order or config["interaction_order"]),
                        ]
                    )
                if model == "mlp" and tuned.get("hidden_dim") is not None:
                    command.extend(["--hidden-dim", str(tuned["hidden_dim"])])
                if model == "mlp" and tuned.get("dropout") is not None:
                    command.extend(["--dropout", str(tuned["dropout"])])

                run_command(command)

if __name__ == "__main__":
    main()

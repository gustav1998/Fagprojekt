from __future__ import annotations

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
    accelerator: str,
    processed_root: Path,
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
                        "--result-version",
                        f"{dataset}_seed{seed}",
                    ]
                    run_command(command)
                    continue

                config = DEFAULT_TRAINING_CONFIGS[model]
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
                    str(learning_rate or config["learning_rate"]),
                    "--patience",
                    str(patience),
                    "--monitor",
                    monitor,
                ]

                if not early_stopping: # checks for early stopping and adds the appropriate flag to the command
                    command.append("--disable-early-stopping")
                if monitor_mode is not None: # checks for monitor mode and adds the appropriate flag to the command
                    command.extend(["--monitor-mode", monitor_mode])
                if model in {"cpd", "tt", "tr"}:
                    command.extend(["--rank", str(rank or config["rank"])]) # checks if the model is one of the factorization models and adds the appropriate rank flag to the command

                if model == "mba":
                    command.extend(
                        [
                            "--interaction-order",
                            str(
                                interaction_order
                                or config["interaction_order"]
                            ),
                        ]
                    ) # checks if the model is mba and adds the appropriate interaction order flag to the command

                run_command(command) # runs the training command for the current model, dataset, and seed

if __name__ == "__main__":
    main()

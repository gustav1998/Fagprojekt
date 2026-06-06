from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from src.data.dataset_configs import DATASET_CONFIGS
from src.train import DEFAULT_TRAINING_CONFIGS


MODELS = ["lr", "mlp", "cpd", "tt", "tr"]


def run_command(command: list[str]) -> None:
    """Run one subprocess command and stop on failure."""
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
    default=Path("data/processed"),
    show_default=True,
)
@click.option("--epochs", type=int, default=None)
@click.option("--learning-rate", type=float, default=None)
@click.option("--rank", type=int, default=None)
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
) -> None:
    """Run preprocessing and training for multiple seeds."""
    selected_datasets = datasets or tuple(sorted(DATASET_CONFIGS))
    selected_models = models or tuple(MODELS)

    for seed in seeds:
        processed_dir = processed_root / f"seed_{seed}"

        for dataset in selected_datasets:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.data.make_dataset",
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

            for model in selected_models:
                config = DEFAULT_TRAINING_CONFIGS[model]
                command = [
                    sys.executable,
                    "-m",
                    "src.train",
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
                ]

                if model in {"cpd", "tt", "tr"}:
                    command.extend(["--rank", str(rank or config["rank"])])

                run_command(command)


if __name__ == "__main__":
    main()

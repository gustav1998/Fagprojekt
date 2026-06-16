"""Optional Optuna hyperparameter tuning for one dataset and model."""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import combinations
from math import prod
from pathlib import Path

import click
import pandas as pd

from src.data_pipeline.dataset_configs import DATASET_CONFIGS
from src.training.run_experiments import MODELS
from src.training.train import DEFAULT_TRAINING_CONFIGS, infer_monitor_mode


def import_optuna():
    """Import Optuna with a clear setup message if it is missing."""
    try:
        import optuna
    except ImportError as error:
        raise SystemExit(
            "Optuna is required for Bayesian tuning. Run `uv sync` first."
        ) from error

    return optuna


def run_command(command: list[str]) -> None:
    """Run a subprocess command and stop on failure."""
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def count_mba_parameters(
    feature_dims: list[int],
    num_classes: int,
    interaction_order: int,
) -> int:
    """Count MBA parameters up to one interaction order."""
    total = num_classes
    for order in range(1, interaction_order + 1):
        for interaction in combinations(feature_dims, order):
            total += num_classes * prod(interaction)
    return total


def allowed_mba_orders(
    metadata_path: Path,
    max_parameters: int,
    max_order: int,
) -> list[int]:
    """Return MBA orders that fit under the parameter budget."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_dims = metadata["cardinalities"]
    num_classes = len(metadata["target_mapping"])
    max_order = min(max_order, len(feature_dims))

    return [
        order
        for order in range(1, max_order + 1)
        if count_mba_parameters(feature_dims, num_classes, order)
        <= max_parameters
    ]


def suggest_params(
    trial,
    model: str,
    mba_orders: list[int] | None = None,
) -> dict[str, int | float | str | None]:
    """Suggest model-specific hyperparameters for one Optuna trial."""
    if model == "rf":
        return {
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 5, 10, 15, 20, 30]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5]
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16),
        }

    params: dict[str, int | float] = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            5e-2,
            log=True,
        )
    }

    if model == "mlp":
        params["hidden_dim"] = trial.suggest_categorical(
            "hidden_dim",
            [64, 128, 256, 512],
        )
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    elif model in {"cpd", "tt", "tr"}:
        params["rank"] = trial.suggest_categorical(
            "rank",
            [4, 8, 16, 32, 64],
        )
    elif model == "mba":
        if not mba_orders:
            raise ValueError("No MBA interaction orders fit the parameter budget.")
        params["interaction_order"] = trial.suggest_categorical(
            "interaction_order",
            mba_orders,
        )

    return params


def read_trial_score(
    result_dir: Path,
    metric: str,
    mode: str,
) -> float:
    """Read the best validation score for one trial."""
    metadata_path = result_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        score = metadata.get("best_model_score")
        if score is not None:
            return float(score)

    metrics = pd.read_csv(result_dir / "metrics.csv")
    values = metrics[metric].dropna()
    if values.empty:
        raise ValueError(f"No values for {metric} in {result_dir}")

    if mode == "min":
        return float(values.min())
    return float(values.max())


@click.command()
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    type=click.Choice(sorted(DATASET_CONFIGS.keys())),
)
@click.option("--model", "models", multiple=True, type=click.Choice(MODELS))
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--trials", type=int, default=25, show_default=True)
@click.option("--epochs", type=int, default=None)
@click.option("--patience", type=int, default=15, show_default=True)
@click.option("--batch-size", type=int, default=512, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option(
    "--max-mba-parameters",
    type=int,
    default=5_000_000,
    show_default=True,
    help="Maximum explicit MBA parameters allowed during tuning.",
)
@click.option(
    "--max-mba-order",
    type=int,
    default=3,
    show_default=True,
    help="Maximum MBA interaction order considered during tuning.",
)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cpu", "gpu"]),
    default="cpu",
    show_default=True,
)
@click.option(
    "--metric",
    type=str,
    default="val_loss",
    show_default=True,
    help="Validation metric to optimize.",
)
@click.option(
    "--metric-mode",
    type=click.Choice(["min", "max"]),
    default=None,
    help="Whether the metric should be minimized or maximized.",
)
@click.option(
    "--processed-root",
    type=click.Path(path_type=Path),
    default=Path("src/data_pipeline/data/processed"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/tuning"),
    show_default=True,
)
@click.option(
    "--skip-preprocessing",
    is_flag=True,
    default=False,
    help="Skip make_dataset calls and assume processed data already exists.",
)
def main(
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seed: int,
    trials: int,
    epochs: int | None,
    patience: int,
    batch_size: int,
    num_workers: int,
    max_mba_parameters: int,
    max_mba_order: int,
    accelerator: str,
    metric: str,
    metric_mode: str | None,
    processed_root: Path,
    output_dir: Path,
    skip_preprocessing: bool,
) -> None:
    """Tune models on datasets using validation performance."""
    optuna = import_optuna()
    selected_datasets = datasets or tuple(sorted(DATASET_CONFIGS.keys()))
    selected_models = models or tuple(MODELS)
    processed_dir = processed_root
    for mi, model in enumerate(selected_models):
        print(
            f"\n{'='*60}\n"
            f"MODEL {mi+1}/{len(selected_models)}: {model.upper()}\n"
            f"{'='*60}",
            flush=True,
        )
        effective_metric = (
            "val_balanced_acc" if model == "rf" and metric == "val_loss" else metric
        )
        effective_mode = metric_mode or infer_monitor_mode(effective_metric)
        effective_direction = "minimize" if effective_mode == "min" else "maximize"
        selected_epochs = (
            None
            if model == "rf"
            else epochs or DEFAULT_TRAINING_CONFIGS[model]["epochs"]
        )
        for di, dataset in enumerate(selected_datasets):
            print(
                f"\n  [{model.upper()}] dataset {di+1}/{len(selected_datasets)}: {dataset}",
                flush=True,
            )
            if not skip_preprocessing:
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

            mba_orders = None
            if model == "mba":
                metadata_path = processed_dir / f"{dataset}_tensor_metadata.json"
                mba_orders = allowed_mba_orders(
                    metadata_path=metadata_path,
                    max_parameters=max_mba_parameters,
                    max_order=max_mba_order,
                )
                if not mba_orders:
                    raise click.ClickException(
                        "No MBA interaction order fits "
                        f"--max-mba-parameters={max_mba_parameters} "
                        f"for dataset {dataset}."
                    )
                print(
                    "Allowed MBA interaction orders for "
                    f"{dataset}: {mba_orders}",
                    flush=True,
                )

            study = optuna.create_study(direction=effective_direction)

            def objective(trial) -> float:
                params = suggest_params(trial, model, mba_orders)
                result_version = (
                    f"tuning_{dataset}_{model}_seed{seed}_trial{trial.number}"
                )

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
                        result_version,
                        "--skip-test",
                        "--max-features",
                        str(params["max_features"]),
                        "--min-samples-leaf",
                        str(params["min_samples_leaf"]),
                    ]
                    if params["max_depth"] is not None:
                        command.extend(["--max-depth", str(params["max_depth"])])
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "src.training.train",
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--processed-dir",
                        str(processed_dir),
                        "--seed",
                        str(seed),
                        "--result-version",
                        result_version,
                        "--epochs",
                        str(selected_epochs),
                        "--batch-size",
                        str(batch_size),
                        "--num-workers",
                        str(num_workers),
                        "--accelerator",
                        accelerator,
                        "--learning-rate",
                        str(params["learning_rate"]),
                        "--patience",
                        str(patience),
                        "--monitor",
                        effective_metric,
                        "--monitor-mode",
                        effective_mode,
                        "--skip-test",
                    ]

                    if "rank" in params:
                        command.extend(["--rank", str(params["rank"])])
                    if "interaction_order" in params:
                        command.extend(
                            ["--interaction-order", str(params["interaction_order"])]
                        )
                    if "hidden_dim" in params:
                        command.extend(["--hidden-dim", str(params["hidden_dim"])])
                    if "dropout" in params:
                        command.extend(["--dropout", str(params["dropout"])])

                run_command(command)
                result_dir = (
                    Path("src/summary_results/results") / model / result_version
                )
                return read_trial_score(result_dir, effective_metric, effective_mode)

            study.optimize(objective, n_trials=trials)

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{model}_{dataset}_seed{seed}_best_params.json"
            output = {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "metric": effective_metric,
                "metric_mode": effective_mode,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trials": trials,
                "epochs": selected_epochs,
                "patience": patience if model != "rf" else None,
                "batch_size": batch_size if model != "rf" else None,
                "num_workers": num_workers if model != "rf" else None,
                "max_mba_parameters": (
                    max_mba_parameters if model == "mba" else None
                ),
                "max_mba_order": max_mba_order if model == "mba" else None,
                "allowed_mba_orders": mba_orders if model == "mba" else None,
            }
            output_path.write_text(
                json.dumps(output, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved {output_path}")
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Optional Optuna hyperparameter tuning for one dataset and model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd

from src.data.dataset_configs import DATASET_CONFIGS
from src.run_experiments import MODELS
from src.train import DEFAULT_TRAINING_CONFIGS, infer_monitor_mode


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


def suggest_params(trial, model: str) -> dict[str, int | float]:
    """Suggest model-specific hyperparameters for one Optuna trial."""
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
        params["interaction_order"] = trial.suggest_int(
            "interaction_order",
            1,
            4,
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
    type=click.Choice(sorted(DATASET_CONFIGS.keys())),
    required=True,
)
@click.option("--model", type=click.Choice(MODELS), required=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--trials", type=int, default=25, show_default=True)
@click.option("--epochs", type=int, default=None)
@click.option("--patience", type=int, default=15, show_default=True)
@click.option("--batch-size", type=int, default=512, show_default=True)
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
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results/tuning"),
    show_default=True,
)
def main(
    dataset: str,
    model: str,
    seed: int,
    trials: int,
    epochs: int | None,
    patience: int,
    batch_size: int,
    accelerator: str,
    metric: str,
    metric_mode: str | None,
    processed_root: Path,
    output_dir: Path,
) -> None:
    """Tune one model on one dataset using validation performance."""
    optuna = import_optuna()
    mode = metric_mode or infer_monitor_mode(metric)
    direction = "minimize" if mode == "min" else "maximize"
    selected_epochs = epochs or DEFAULT_TRAINING_CONFIGS[model]["epochs"]
    processed_dir = processed_root / f"seed_{seed}"

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

    study = optuna.create_study(direction=direction)

    def objective(trial) -> float:
        params = suggest_params(trial, model)
        result_version = (
            f"tuning_{dataset}_{model}_seed{seed}_trial{trial.number}"
        )
        command = [
            sys.executable,
            "-m",
            "src.train",
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
            "--accelerator",
            accelerator,
            "--learning-rate",
            str(params["learning_rate"]),
            "--patience",
            str(patience),
            "--monitor",
            metric,
            "--monitor-mode",
            mode,
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
        result_dir = Path("results") / model / result_version
        return read_trial_score(result_dir, metric, mode)

    study.optimize(objective, n_trials=trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model}_{dataset}_seed{seed}_best_params.json"
    output = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "metric": metric,
        "metric_mode": mode,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": trials,
        "epochs": selected_epochs,
        "patience": patience,
        "batch_size": batch_size,
    }
    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

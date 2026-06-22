"""Analysis and visualization helpers for K-fold experiment results.

The functions in this file read the training logs in
``src/summary_results/results`` and write plots/tables to ``results`` by
default. They are intentionally kept as plain functions so they can be reused
or adjusted from notebooks later.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


MODELS = ("lr", "mlp", "cpd", "mba", "tt", "tr", "rf")
FOLD_PATTERN = re.compile(
    r"(?P<dataset>.+)_fold(?P<fold>\d+)_seed(?P<seed>-?\d+)$"
)


# Shared file helpers

def ensure_output_dir(output_dir: Path) -> Path:
    """Create the output folder if needed and return it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plot(path: Path) -> None:
    """Save the current matplotlib figure with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close()


def read_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict if it is missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_final_metric_row(
    metrics: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:
    """Return the last row with at least one useful metric value."""
    available = [column for column in columns if column in metrics.columns]
    if not available:
        return metrics.iloc[-1]

    rows = metrics[metrics[available].notna().any(axis=1)]
    if rows.empty:
        return metrics.iloc[-1]
    return rows.iloc[-1]


def parse_fold_version(version: str) -> tuple[str | None, int | None, int | None]:
    """Extract dataset, fold, and seed from a result folder name."""
    match = FOLD_PATTERN.fullmatch(version)
    if match is None:
        return None, None, None
    return (
        match.group("dataset"),
        int(match.group("fold")),
        int(match.group("seed")),
    )


# Step 1: Read result files

def read_fold_results(
    results_root: Path = Path("src/summary_results/results"),
    models: Iterable[str] = MODELS,
) -> pd.DataFrame:
    """Read one final metric row per model, dataset, fold, and seed."""
    rows: list[dict] = []
    for model in models:
        model_dir = results_root / model
        if not model_dir.exists():
            continue

        for metrics_path in sorted(model_dir.glob("*/metrics.csv")):
            dataset, fold, seed = parse_fold_version(metrics_path.parent.name)
            if dataset is None:
                continue

            metrics = pd.read_csv(metrics_path)
            final = get_final_metric_row(
                metrics,
                [
                    "test_acc",
                    "test_loss",
                    "test_balanced_acc",
                    "test_macro_f1",
                    "val_acc",
                    "val_loss",
                ],
            )
            metadata = read_json(metrics_path.parent / "run_metadata.json")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "fold": fold,
                    "seed": seed,
                    "version": metrics_path.parent.name,
                    "test_acc": final.get("test_acc"),
                    "test_loss": final.get("test_loss"),
                    "test_balanced_acc": final.get("test_balanced_acc"),
                    "test_macro_f1": final.get("test_macro_f1"),
                    "test_weighted_f1": final.get("test_weighted_f1"),
                    "val_acc": final.get("val_acc"),
                    "val_loss": final.get("val_loss"),
                    "fit_seconds": metadata.get("fit_seconds"),
                    "test_seconds": metadata.get("test_seconds"),
                    "trainable_parameters": metadata.get("trainable_parameters"),
                    "learning_rate": metadata.get("learning_rate"),
                    "rank": metadata.get("rank"),
                    "interaction_order": metadata.get("interaction_order"),
                    "epochs": metadata.get("epochs"),
                    "stopped_epoch": metadata.get("stopped_epoch"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    numeric_columns = [
        "fold",
        "seed",
        "test_acc",
        "test_loss",
        "test_balanced_acc",
        "test_macro_f1",
        "test_weighted_f1",
        "val_acc",
        "val_loss",
        "fit_seconds",
        "test_seconds",
        "trainable_parameters",
        "learning_rate",
        "rank",
        "interaction_order",
        "epochs",
        "stopped_epoch",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values(["dataset", "model", "fold"]).reset_index(drop=True)


def read_tuning_results(
    dataset: str,
    model: str,
    results_root: Path = Path("src/summary_results/results"),
) -> pd.DataFrame:
    """Load validation curves for every tuning combo for one dataset/model."""
    model_dir = results_root / model
    combo_pattern = re.compile(
        rf"^tuning_{re.escape(dataset)}_{re.escape(model)}_combo(?P<combo>\d+)$"
    )
    rows: list[dict] = []

    for result_dir in sorted(model_dir.glob(f"tuning_{dataset}_{model}_combo*")):
        match = combo_pattern.fullmatch(result_dir.name)
        if match is None:
            continue
        metrics_path = result_dir / "metrics.csv"
        if not metrics_path.exists():
            continue

        combo = int(match.group("combo"))
        metadata = read_json(result_dir / "run_metadata.json")
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            if not any(pd.notna(row.get(col)) for col in ["val_loss", "val_acc"]):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "combo": combo,
                    "epoch": row.get("epoch"),
                    "step": row.get("step"),
                    "val_loss": row.get("val_loss"),
                    "val_acc": row.get("val_acc"),
                    "val_balanced_acc": row.get("val_balanced_acc"),
                    "val_macro_f1": row.get("val_macro_f1"),
                    "learning_rate": metadata.get("learning_rate"),
                    "rank": metadata.get("rank"),
                    "interaction_order": metadata.get("interaction_order"),
                    "hidden_dim": metadata.get("hidden_dim"),
                    "dropout": metadata.get("dropout"),
                    "best_model_score": metadata.get("best_model_score"),
                    "fit_seconds": metadata.get("fit_seconds"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["combo", "epoch", "step"]).reset_index(drop=True)


# Step 2: Make optional plots

def plot_tuning_curves(
    dataset: str,
    model: str,
    metric: str = "val_loss",
    results_root: Path = Path("src/summary_results/results"),
    output_dir: Path = Path("results"),
) -> pd.DataFrame:
    """Plot tuning curves for every grid-search combination."""
    data = read_tuning_results(dataset, model, results_root)
    if data.empty:
        raise ValueError(f"No tuning data found for {model} on {dataset}.")
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' is not available in tuning data.")

    ensure_output_dir(output_dir)
    table_path = output_dir / f"tuning_{dataset}_{model}.csv"
    data.to_csv(table_path, index=False)

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    for combo, combo_data in data.groupby("combo"):
        combo_data = combo_data.dropna(subset=[metric])
        if combo_data.empty:
            continue
        x_axis = combo_data["epoch"].fillna(combo_data["step"])
        label_parts = [f"combo {combo}"]
        for param in [
            "learning_rate",
            "rank",
            "interaction_order",
            "hidden_dim",
            "dropout",
        ]:
            values = combo_data[param].dropna()
            value = values.iloc[0] if not values.empty else None
            if value is not None:
                label_parts.append(f"{param}={value:g}")
        ax.plot(
            x_axis,
            combo_data[metric],
            linewidth=1.5,
            alpha=0.85,
            label=", ".join(label_parts),
        )

    ax.set_title(f"Tuning {metric}: {model.upper()} on {dataset}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_plot(output_dir / f"tuning_{dataset}_{model}_{metric}.png")
    return data


def plot_dataset_summary(
    dataset: str,
    results_root: Path = Path("src/summary_results/results"),
    output_dir: Path = Path("results"),
) -> pd.DataFrame:
    """Plot all model accuracies and losses for one dataset across folds."""
    data = read_fold_results(results_root)
    data = data[data["dataset"] == dataset].copy()
    if data.empty:
        raise ValueError(f"No fold results found for dataset '{dataset}'.")

    summary = (
        data.groupby("model", as_index=False)
        .agg(
            accuracy_mean=("test_acc", "mean"),
            accuracy_std=("test_acc", "std"),
            loss_mean=("test_loss", "mean"),
            loss_std=("test_loss", "std"),
            folds=("fold", "nunique"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )

    ensure_output_dir(output_dir)
    summary.to_csv(output_dir / f"{dataset}_model_accuracy_loss.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    axes[0].bar(
        summary["model"],
        summary["accuracy_mean"],
        yerr=summary["accuracy_std"],
        capsize=4,
    )
    axes[0].set_title(f"{dataset}: accuracy by model")
    axes[0].set_ylabel("Mean test accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)

    loss_data = summary.dropna(subset=["loss_mean"])
    axes[1].bar(
        loss_data["model"],
        loss_data["loss_mean"],
        yerr=loss_data["loss_std"],
        capsize=4,
        color="#b66a50",
    )
    axes[1].set_title(f"{dataset}: loss by model")
    axes[1].set_ylabel("Mean test loss")
    axes[1].grid(axis="y", alpha=0.25)

    save_plot(output_dir / f"{dataset}_accuracy_loss_by_model.png")
    return summary


# Step 3: Summarize result tables

def summarize_accuracy_and_loss(
    results_root: Path = Path("src/summary_results/results"),
    output_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average accuracy/loss across folds for all datasets and models."""
    data = read_fold_results(results_root)
    if data.empty:
        raise ValueError("No fold results found.")

    dataset_model = (
        data.groupby(["dataset", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            accuracy_mean=("test_acc", "mean"),
            accuracy_std=("test_acc", "std"),
            loss_mean=("test_loss", "mean"),
            loss_std=("test_loss", "std"),
            balanced_acc_mean=("test_balanced_acc", "mean"),
            macro_f1_mean=("test_macro_f1", "mean"),
        )
        .sort_values(["dataset", "model"])
    )

    overall = (
        dataset_model.groupby("model", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_std_across_datasets=("accuracy_mean", "std"),
            loss_mean=("loss_mean", "mean"),
            loss_std_across_datasets=("loss_mean", "std"),
            balanced_acc_mean=("balanced_acc_mean", "mean"),
            macro_f1_mean=("macro_f1_mean", "mean"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )

    ensure_output_dir(output_dir)
    dataset_model.to_csv(
        output_dir / "accuracy_loss_by_dataset_model.csv",
        index=False,
    )
    overall.to_csv(output_dir / "accuracy_loss_overall_by_model.csv", index=False)
    return dataset_model, overall


def summarize_training_time(
    results_root: Path = Path("src/summary_results/results"),
    output_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average fit/test time across folds for all datasets and models."""
    data = read_fold_results(results_root)
    if data.empty:
        raise ValueError("No fold results found.")

    dataset_model = (
        data.groupby(["dataset", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", "std"),
            test_seconds_mean=("test_seconds", "mean"),
            test_seconds_std=("test_seconds", "std"),
        )
        .sort_values(["dataset", "model"])
    )

    overall = (
        dataset_model.groupby("model", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            fit_seconds_mean=("fit_seconds_mean", "mean"),
            fit_seconds_std_across_datasets=("fit_seconds_mean", "std"),
            test_seconds_mean=("test_seconds_mean", "mean"),
            test_seconds_std_across_datasets=("test_seconds_mean", "std"),
        )
        .sort_values("fit_seconds_mean")
    )

    ensure_output_dir(output_dir)
    dataset_model.to_csv(output_dir / "timing_by_dataset_model.csv", index=False)
    overall.to_csv(output_dir / "timing_overall_by_model.csv", index=False)
    return dataset_model, overall


# Step 4: Run statistical tests

def build_metric_matrix(
    data: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Build a complete dataset x model metric matrix."""
    dataset_model = data.groupby(["dataset", "model"], as_index=False)[metric].mean()
    matrix = dataset_model.pivot(index="dataset", columns="model", values=metric)
    matrix = matrix.dropna(axis=0, how="any")
    if not higher_is_better:
        matrix = -matrix
    return matrix


def run_friedman_nemenyi_tests(
    metric: str = "test_acc",
    higher_is_better: bool = True,
    results_root: Path = Path("src/summary_results/results"),
    output_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Friedman and Nemenyi tests on model scores across datasets.

    The test uses one score per dataset/model, averaged across folds first.
    """
    data = read_fold_results(results_root)
    if data.empty:
        raise ValueError("No fold results found.")
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' is not available.")

    matrix = build_metric_matrix(data, metric, higher_is_better)
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError(
            "Friedman/Nemenyi needs at least 2 complete datasets and 3 models."
        )

    statistic, p_value = friedmanchisquare(
        *[matrix[model].to_numpy() for model in matrix.columns]
    )
    friedman = pd.DataFrame(
        [
            {
                "metric": metric,
                "higher_is_better": higher_is_better,
                "datasets": matrix.shape[0],
                "models": matrix.shape[1],
                "friedman_statistic": statistic,
                "friedman_p_value": p_value,
            }
        ]
    )

    ranks = matrix.apply(
        lambda row: rankdata(-row.to_numpy(), method="average"),
        axis=1,
        result_type="expand",
    )
    ranks.columns = matrix.columns
    average_ranks = (
        ranks.mean(axis=0)
        .rename("average_rank")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values("average_rank")
    )

    n_datasets = matrix.shape[0]
    n_models = matrix.shape[1]
    scale = math.sqrt(6 * n_datasets / (n_models * (n_models + 1)))
    nemenyi = pd.DataFrame(index=matrix.columns, columns=matrix.columns, dtype=float)
    rank_values = average_ranks.set_index("model")["average_rank"]
    for model_a in matrix.columns:
        for model_b in matrix.columns:
            diff = abs(rank_values[model_a] - rank_values[model_b])
            q_stat = diff * scale * math.sqrt(2)
            nemenyi.loc[model_a, model_b] = studentized_range.sf(
                q_stat,
                n_models,
                np.inf,
            )

    ensure_output_dir(output_dir)
    friedman.to_csv(output_dir / f"friedman_{metric}.csv", index=False)
    average_ranks.to_csv(output_dir / f"average_ranks_{metric}.csv", index=False)
    nemenyi.to_csv(output_dir / f"nemenyi_p_values_{metric}.csv")
    matrix.to_csv(output_dir / f"statistical_test_matrix_{metric}.csv")
    return friedman, average_ranks, nemenyi


# Step 5: Command-line entry points

def run_all(
    dataset: str | None,
    model: str | None,
    tuning_metric: str,
    stats_metric: str,
    results_root: Path,
    output_dir: Path,
) -> None:
    """Run all available tables and optional dataset/model plots."""
    ensure_output_dir(output_dir)
    if dataset and model:
        plot_tuning_curves(dataset, model, tuning_metric, results_root, output_dir)
    if dataset:
        plot_dataset_summary(dataset, results_root, output_dir)
    summarize_accuracy_and_loss(results_root, output_dir)
    summarize_training_time(results_root, output_dir)
    run_friedman_nemenyi_tests(stats_metric, True, results_root, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots, tables, and statistical tests from experiment results."
    )
    parser.add_argument(
        "command",
        choices=["all", "tuning", "dataset", "tables", "timing", "stats"],
        help="Which analysis to run.",
    )
    parser.add_argument("--dataset", help="Dataset name for dataset/tuning plots.")
    parser.add_argument("--model", choices=MODELS, help="Model name for tuning plots.")
    parser.add_argument("--tuning-metric", default="val_loss")
    parser.add_argument("--stats-metric", default="test_acc")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Use this for statistical tests on loss or timing metrics.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("src/summary_results/results"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "tuning":
        if not args.dataset or not args.model:
            raise SystemExit("--dataset and --model are required for tuning plots.")
        plot_tuning_curves(
            args.dataset,
            args.model,
            args.tuning_metric,
            args.results_root,
            args.output_dir,
        )
    elif args.command == "dataset":
        if not args.dataset:
            raise SystemExit("--dataset is required for dataset plots.")
        plot_dataset_summary(args.dataset, args.results_root, args.output_dir)
    elif args.command == "tables":
        summarize_accuracy_and_loss(args.results_root, args.output_dir)
    elif args.command == "timing":
        summarize_training_time(args.results_root, args.output_dir)
    elif args.command == "stats":
        run_friedman_nemenyi_tests(
            args.stats_metric,
            not args.lower_is_better,
            args.results_root,
            args.output_dir,
        )
    elif args.command == "all":
        run_all(
            args.dataset,
            args.model,
            args.tuning_metric,
            args.stats_metric,
            args.results_root,
            args.output_dir,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from src.data_pipeline.dataset_configs import DATASET_CONFIGS

try:
    import scikit_posthocs as sp
except ImportError:  # Optional dependency; Friedman still works without it.
    sp = None


MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tt3", "tr", "rf"]
N_FOLDS = 5
RESULTS_DIR = Path("src/summary_results/results")
OUTPUT_DIR = Path("src/visualization/postprocessing")
DATASET_REPORTS_DIR = Path("src/data_pipeline/data/reports")
EXCLUDED_DATASETS = {"lenses"}
EXCLUDED_PREFIXES = ("monk",)

TRAINING_METRICS = [
    "train_acc",
    "train_balanced_acc",
    "train_loss",
    "train_macro_f1",
    "val_acc",
    "val_balanced_acc",
    "val_loss",
    "val_macro_f1",
]
TEST_METRICS = [
    "test_acc",
    "test_balanced_acc",
    "test_loss",
    "test_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_weighted_f1",
]
TIME_COLUMNS = ["fit_seconds", "test_seconds", "total_seconds"]


def is_excluded_dataset(dataset: str) -> bool:
    return dataset in EXCLUDED_DATASETS or dataset.startswith(EXCLUDED_PREFIXES)


def configured_datasets(include_excluded: bool = False) -> list[str]:
    datasets = sorted(DATASET_CONFIGS)
    if include_excluded:
        return datasets
    return [dataset for dataset in datasets if not is_excluded_dataset(dataset)]


def parse_run_name(model: str, run_name: str) -> dict[str, object] | None:
    tuning_match = re.fullmatch(
        rf"tuning_(?P<dataset>.+)_{re.escape(model)}_combo(?P<combo>\d+)",
        run_name,
    )
    if tuning_match:
        return {
            "dataset": tuning_match.group("dataset"),
            "model": model,
            "kind": "tuning",
            "fold": pd.NA,
            "seed": pd.NA,
            "combo": int(tuning_match.group("combo")),
            "run_name": run_name,
        }

    fold_match = re.fullmatch(
        r"(?P<dataset>.+)_fold(?P<fold>\d+)_seed(?P<seed>-?\d+)",
        run_name,
    )
    if fold_match:
        return {
            "dataset": fold_match.group("dataset"),
            "model": model,
            "kind": "fold",
            "fold": int(fold_match.group("fold")),
            "seed": int(fold_match.group("seed")),
            "combo": pd.NA,
            "run_name": run_name,
        }

    return None


def collapse_lightning_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse Lightning's sparse alternating metric rows into one row per epoch."""
    if {"epoch", "step"}.issubset(metrics.columns):
        metrics = metrics.groupby(["epoch", "step"], as_index=False).first()
    return metrics


def read_metrics_file(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    metrics = collapse_lightning_metrics(metrics)
    if "epoch" in metrics.columns:
        metrics["epoch"] = pd.to_numeric(metrics["epoch"], errors="coerce")
    if "step" in metrics.columns:
        metrics["step"] = pd.to_numeric(metrics["step"], errors="coerce")
    return metrics


def iter_result_runs(
    results_dir: Path = RESULTS_DIR,
    models: Iterable[str] = MODELS,
    include_excluded: bool = False,
):
    for model in models:
        model_dir = results_dir / model
        if not model_dir.exists():
            continue

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            parsed = parse_run_name(model, run_dir.name)
            if parsed is None:
                continue
            dataset = str(parsed["dataset"])
            if not include_excluded and is_excluded_dataset(dataset):
                continue
            yield model, run_dir, parsed


def load_metrics(
    results_dir: Path = RESULTS_DIR,
    models: Iterable[str] = MODELS,
    dataset: str | None = None,
    include_excluded: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for _, run_dir, parsed in iter_result_runs(
        results_dir=results_dir,
        models=models,
        include_excluded=include_excluded,
    ):
        if dataset is not None and parsed["dataset"] != dataset:
            continue
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue

        frame = read_metrics_file(metrics_path)
        for key, value in parsed.items():
            frame[key] = value
        frame["metrics_path"] = str(metrics_path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_metadata(
    results_dir: Path = RESULTS_DIR,
    models: Iterable[str] = MODELS,
    dataset: str | None = None,
    include_excluded: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, run_dir, parsed in iter_result_runs(
        results_dir=results_dir,
        models=models,
        include_excluded=include_excluded,
    ):
        if dataset is not None and parsed["dataset"] != dataset:
            continue
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = {**parsed, **metadata}
        row["metadata_path"] = str(metadata_path)
        rows.append(row)

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        return metadata

    for column in ["fit_seconds", "test_seconds"]:
        if column in metadata.columns:
            metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata["total_seconds"] = (
        metadata.get("fit_seconds", pd.Series(dtype=float)).fillna(0)
        + metadata.get("test_seconds", pd.Series(dtype=float)).fillna(0)
    )
    return metadata


def available_metric_columns(metrics: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [
        column
        for column in candidates
        if column in metrics.columns and metrics[column].notna().any()
    ]


def final_non_null_by_run(
    metrics: pd.DataFrame,
    metric_columns: Iterable[str] = TEST_METRICS,
) -> pd.DataFrame:
    """Return one row per fold run, taking each metric's last non-null value."""
    fold_metrics = metrics[metrics["kind"] == "fold"].copy()
    if fold_metrics.empty:
        return pd.DataFrame()

    id_columns = ["dataset", "model", "fold", "seed", "run_name"]
    metric_columns = [column for column in metric_columns if column in fold_metrics.columns]
    rows: list[dict[str, object]] = []

    for keys, group in fold_metrics.groupby(id_columns, dropna=False):
        group = group.sort_values(["epoch", "step"], na_position="last")
        row = dict(zip(id_columns, keys))
        for metric in metric_columns:
            values = group[metric].dropna()
            row[metric] = values.iloc[-1] if not values.empty else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def completeness_table(
    results_dir: Path = RESULTS_DIR,
    models: Iterable[str] = MODELS,
    include_excluded: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model in models:
        model_runs = list(
            iter_result_runs(
                results_dir=results_dir,
                models=[model],
                include_excluded=include_excluded,
            )
        )
        datasets = sorted({str(parsed["dataset"]) for _, _, parsed in model_runs})
        for dataset_name in datasets:
            folds = set()
            tuning_count = 0
            fold_metrics_count = 0
            tuning_metrics_count = 0

            for _, run_dir, parsed in model_runs:
                if parsed["dataset"] != dataset_name:
                    continue
                metrics_exists = (run_dir / "metrics.csv").exists()
                if parsed["kind"] == "fold":
                    folds.add(int(parsed["fold"]))
                    if metrics_exists:
                        fold_metrics_count += 1
                elif parsed["kind"] == "tuning":
                    tuning_count += 1
                    if metrics_exists:
                        tuning_metrics_count += 1

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "folds": ",".join(str(fold) for fold in sorted(folds)),
                    "fold_count": len(folds),
                    "fold_metrics_count": fold_metrics_count,
                    "tuning_count": tuning_count,
                    "tuning_metrics_count": tuning_metrics_count,
                    "has_all_folds": folds == set(range(1, N_FOLDS + 1)),
                    "has_tuning": tuning_metrics_count > 0,
                    "complete": folds == set(range(1, N_FOLDS + 1))
                    and tuning_metrics_count > 0,
                }
            )

    return pd.DataFrame(rows)


def complete_pairs_only(metrics: pd.DataFrame, completeness: pd.DataFrame) -> pd.DataFrame:
    complete = completeness[completeness["complete"]][["dataset", "model"]]
    return metrics.merge(complete, on=["dataset", "model"], how="inner")


def maybe_scaled_epoch(group: pd.DataFrame, scale_epochs: bool) -> pd.Series:
    epochs = group["epoch"].astype(float)
    max_epoch = epochs.max()
    if scale_epochs and pd.notna(max_epoch) and max_epoch > 0:
        return epochs / max_epoch
    return epochs


def plot_tuning_combinations(
    dataset: str,
    model: str,
    metric: str | None = None,
    results_dir: Path = RESULTS_DIR,
    output_dir: Path = OUTPUT_DIR,
    scale_epochs: bool = False,
) -> Path:
    metrics = load_metrics(results_dir=results_dir, models=[model], dataset=dataset)
    tuning = metrics[metrics["kind"] == "tuning"].copy()
    if tuning.empty:
        raise ValueError(f"No tuning metrics found for {dataset}/{model}.")

    if metric is None:
        metric = "val_balanced_acc" if model == "rf" else "val_loss"
    if metric not in tuning.columns or tuning[metric].notna().sum() == 0:
        available = available_metric_columns(tuning, TRAINING_METRICS + TEST_METRICS)
        raise ValueError(
            f"Metric {metric!r} is unavailable for {dataset}/{model}. "
            f"Available metrics: {available}"
        )

    plot_dir = output_dir / "tuning"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))

    for combo, combo_df in tuning.groupby("combo", dropna=True):
        combo_df = combo_df.dropna(subset=[metric]).sort_values("epoch")
        if combo_df.empty:
            continue
        x = maybe_scaled_epoch(combo_df, scale_epochs)
        plt.plot(x, combo_df[metric], label=f"combo {int(combo)}", alpha=0.75)

    plt.title(f"Tuning {metric}: {dataset} / {model}")
    plt.xlabel("Training progress" if scale_epochs else "Epoch")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    output_path = plot_dir / f"{dataset}_{model}_{metric}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_dataset_accuracy_loss(
    dataset: str,
    results_dir: Path = RESULTS_DIR,
    output_dir: Path = OUTPUT_DIR,
    scale_epochs: bool = False,
    require_complete: bool = True,
) -> Path:
    metrics = load_metrics(results_dir=results_dir, dataset=dataset)
    if metrics.empty:
        raise ValueError(f"No metrics found for {dataset}.")
    if require_complete:
        completeness = completeness_table(results_dir=results_dir)
        metrics = complete_pairs_only(metrics, completeness)

    folds = metrics[metrics["kind"] == "fold"].copy()
    if folds.empty:
        raise ValueError(f"No fold metrics found for {dataset}.")

    plot_dir = output_dir / "datasets"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axis_specs = [("val_acc", axes[0], "Validation accuracy"), ("val_loss", axes[1], "Validation loss")]

    for model in MODELS:
        model_df = folds[folds["model"] == model]
        if model_df.empty:
            continue
        for metric, axis, title in axis_specs:
            if metric not in model_df.columns or model_df[metric].notna().sum() == 0:
                continue

            curves = []
            for _, fold_df in model_df.groupby("fold", dropna=True):
                fold_df = fold_df.dropna(subset=[metric]).sort_values("epoch")
                if fold_df.empty:
                    continue
                curve = pd.DataFrame(
                    {
                        "epoch": maybe_scaled_epoch(fold_df, scale_epochs),
                        metric: fold_df[metric].to_numpy(),
                    }
                )
                curves.append(curve)

            if not curves:
                continue
            mean_curve = pd.concat(curves).groupby("epoch", as_index=False)[metric].mean()
            axis.plot(mean_curve["epoch"], mean_curve[metric], label=model)

    for _, axis, title in axis_specs:
        axis.set_title(f"{dataset}: {title}")
        axis.set_xlabel("Training progress" if scale_epochs else "Epoch")
        axis.grid(alpha=0.3)
        axis.legend()

    output_path = plot_dir / f"{dataset}_accuracy_loss.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def summarize_accuracy_loss(
    results_dir: Path = RESULTS_DIR,
    output_path: Path = OUTPUT_DIR / "accuracy_loss_summary.csv",
    require_complete: bool = True,
) -> pd.DataFrame:
    metrics = load_metrics(results_dir=results_dir)
    if metrics.empty:
        raise ValueError(f"No metrics found under {results_dir}.")
    if require_complete:
        metrics = complete_pairs_only(metrics, completeness_table(results_dir=results_dir))

    metric_columns = available_metric_columns(metrics, TEST_METRICS)
    finals = final_non_null_by_run(metrics, metric_columns=metric_columns)
    if finals.empty:
        raise ValueError("No fold-level final metrics were found.")

    grouped = (
        finals.groupby(["dataset", "model"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]

    tables = []
    for metric in metric_columns:
        column = f"{metric}_mean"
        if column not in grouped.columns:
            continue
        table = grouped.pivot(index="dataset", columns="model", values=column)
        table = table.reindex(columns=MODELS)
        table.insert(0, "metric", metric)
        table.loc["AVERAGE", "metric"] = metric
        table.loc["AVERAGE", MODELS] = table[MODELS].mean(numeric_only=True)
        tables.append(table.reset_index())

    summary = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def summarize_time_consumption(
    results_dir: Path = RESULTS_DIR,
    output_path: Path = OUTPUT_DIR / "time_summary.csv",
    require_complete: bool = True,
) -> pd.DataFrame:
    metadata = load_metadata(results_dir=results_dir)
    if metadata.empty:
        raise ValueError(f"No run metadata found under {results_dir}.")
    metadata = metadata[metadata["kind"] == "fold"].copy()
    if require_complete:
        metadata = complete_pairs_only(metadata, completeness_table(results_dir=results_dir))

    metric_columns = [column for column in TIME_COLUMNS if column in metadata.columns]
    grouped = metadata.groupby(["dataset", "model"])[metric_columns].mean().reset_index()

    tables = []
    for metric in metric_columns:
        table = grouped.pivot(index="dataset", columns="model", values=metric)
        table = table.reindex(columns=MODELS)
        table.insert(0, "metric", metric)
        table.loc["AVERAGE", "metric"] = metric
        table.loc["AVERAGE", MODELS] = table[MODELS].mean(numeric_only=True)
        tables.append(table.reset_index())

    summary = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def friedman_nemenyi_tests(
    metric: str = "test_acc",
    results_dir: Path = RESULTS_DIR,
    output_dir: Path = OUTPUT_DIR / "statistics",
    require_complete: bool = True,
) -> dict[str, pd.DataFrame]:
    metrics = load_metrics(results_dir=results_dir)
    if require_complete:
        metrics = complete_pairs_only(metrics, completeness_table(results_dir=results_dir))
    finals = final_non_null_by_run(metrics, metric_columns=[metric])
    if finals.empty or metric not in finals:
        raise ValueError(f"No final fold values found for {metric}.")

    scores = (
        finals.groupby(["dataset", "model"])[metric]
        .mean()
        .reset_index()
        .pivot(index="dataset", columns="model", values=metric)
        .reindex(columns=MODELS)
        .dropna(axis=0, how="any")
    )
    if scores.shape[0] < 2 or scores.shape[1] < 2:
        raise ValueError(
            "Friedman test needs at least two complete datasets and two models."
        )

    statistic, p_value = friedmanchisquare(
        *[scores[model].to_numpy() for model in scores.columns]
    )
    friedman = pd.DataFrame(
        [
            {
                "metric": metric,
                "n_datasets": scores.shape[0],
                "n_models": scores.shape[1],
                "statistic": statistic,
                "p_value": p_value,
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / f"{metric}_scores_matrix.csv")
    friedman.to_csv(output_dir / f"{metric}_friedman.csv", index=False)

    results = {"scores": scores, "friedman": friedman}
    if sp is not None:
        nemenyi = sp.posthoc_nemenyi_friedman(scores)
        nemenyi.to_csv(output_dir / f"{metric}_nemenyi.csv")
        results["nemenyi"] = nemenyi
    else:
        note = pd.DataFrame(
            [
                {
                    "message": (
                        "Install scikit-posthocs to compute Nemenyi p-values: "
                        "uv add scikit-posthocs"
                    )
                }
            ]
        )
        note.to_csv(output_dir / f"{metric}_nemenyi_not_run.csv", index=False)
        results["nemenyi_note"] = note

    return results


def calculate_table_metrics(
    results: pd.DataFrame | None = None,
    cardinalities: str | None = None,
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Calculate dataset-level tensor metrics from the generated data reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary_path = DATASET_REPORTS_DIR / "dataset_summary.csv"
    if not dataset_summary_path.exists():
        raise FileNotFoundError(dataset_summary_path)

    results = pd.read_csv(dataset_summary_path)
    cardinalities_df = pd.read_csv(cardinalities) if cardinalities is not None else None
    rows = []

    for dataset_name in results["dataset"].unique():
        dataset_subset = results[results["dataset"] == dataset_name]
        first_representation = dataset_subset["representation"].iloc[0]
        subset = dataset_subset[dataset_subset["representation"] == first_representation]

        total_rows = subset["rows"].sum()
        features = subset["features"].iloc[0] + 1
        classes = subset["classes"].iloc[0]

        tensor_size = None
        if cardinalities_df is not None:
            dataset_cards = cardinalities_df[
                (cardinalities_df["dataset"] == dataset_name)
                & (cardinalities_df["representation"] == "tensor")
            ]
            cardinality_values = pd.to_numeric(
                dataset_cards["cardinality"],
                errors="coerce",
            ).dropna()
            if not cardinality_values.empty:
                tensor_size = int(np.prod(cardinality_values.to_numpy()) * classes)

        if tensor_size is None:
            tensor_size = total_rows * features

        rows.append(
            {
                "dataset": dataset_name,
                "features": features,
                "non_zero_entries": total_rows,
                "tensor_size": tensor_size,
                "sparsity": total_rows / tensor_size if tensor_size > 0 else 0,
                "classes": classes,
            }
        )

    return pd.DataFrame(rows)


def generate_plots(results: pd.DataFrame, output_dir: str | Path = OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not results["dataset"].isin(DATASET_CONFIGS.keys()).any():
        raise ValueError("Dataset names in results do not match configured datasets.")

    metrics = results[["dataset", "non_zero_entries"]].copy()

    plt.figure(figsize=(10, 6))
    metrics.boxplot(column="non_zero_entries", by="dataset")
    plt.title("Box Plot of Non-Zero Entries per Dataset")
    plt.suptitle("")
    plt.xlabel("Dataset")
    plt.ylabel("# Non-Zero Entries")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "non_zero_entries_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    metrics.groupby("dataset")["non_zero_entries"].sum().plot(kind="bar")
    plt.title("Total Non-Zero Entries per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Total Non-Zero Entries")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "non_zero_entries_barplot.png", dpi=200)
    plt.close()


def generate_epoch_plots(
    metrics_path: str | Path = RESULTS_DIR,
    output_dir: str | Path = OUTPUT_DIR,
) -> None:
    metrics = load_metrics(results_dir=Path(metrics_path), include_excluded=True)
    if metrics.empty:
        raise ValueError(f"No metrics found under {metrics_path}.")

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots" / "epoch"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for run_name, run_df in metrics.groupby("run_name"):
        model = run_df["model"].iloc[0]
        dataset = run_df["dataset"].iloc[0]
        available = available_metric_columns(run_df, TRAINING_METRICS + TEST_METRICS)
        if not available:
            continue

        plt.figure(figsize=(10, 6))
        for metric in available:
            values = run_df.dropna(subset=[metric]).sort_values("epoch")
            if values.empty:
                continue
            plt.plot(values["epoch"], values[metric], marker="o", label=metric)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Metrics for {model} - {run_name}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=8)
        dataset_plot_dir = plot_dir / str(dataset)
        dataset_plot_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(dataset_plot_dir / f"{model}_{run_name}.png", dpi=200)
        plt.close()


def save_completeness(
    results_dir: Path = RESULTS_DIR,
    output_path: Path = OUTPUT_DIR / "completeness.csv",
) -> pd.DataFrame:
    completeness = completeness_table(results_dir=results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completeness.to_csv(output_path, index=False)
    return completeness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess experiment results.")
    subparsers = parser.add_subparsers(dest="command")

    completeness = subparsers.add_parser("completeness")
    completeness.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    completeness.add_argument("--output", type=Path, default=OUTPUT_DIR / "completeness.csv")

    tuning = subparsers.add_parser("plot-tuning")
    tuning.add_argument("--dataset", required=True)
    tuning.add_argument("--model", required=True, choices=MODELS)
    tuning.add_argument("--metric", default=None)
    tuning.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    tuning.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    tuning.add_argument("--scale-epochs", action="store_true")

    dataset_plot = subparsers.add_parser("plot-dataset")
    dataset_plot.add_argument("--dataset", required=True)
    dataset_plot.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    dataset_plot.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    dataset_plot.add_argument("--scale-epochs", action="store_true")
    dataset_plot.add_argument("--allow-incomplete", action="store_true")

    accuracy = subparsers.add_parser("summary-accuracy-loss")
    accuracy.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    accuracy.add_argument("--output", type=Path, default=OUTPUT_DIR / "accuracy_loss_summary.csv")
    accuracy.add_argument("--allow-incomplete", action="store_true")

    time_summary = subparsers.add_parser("summary-time")
    time_summary.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    time_summary.add_argument("--output", type=Path, default=OUTPUT_DIR / "time_summary.csv")
    time_summary.add_argument("--allow-incomplete", action="store_true")

    stats = subparsers.add_parser("stats")
    stats.add_argument("--metric", default="test_acc")
    stats.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    stats.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "statistics")
    stats.add_argument("--allow-incomplete", action="store_true")

    table_metrics = subparsers.add_parser("dataset-metrics")
    table_metrics.add_argument("--cardinalities", default=None)
    table_metrics.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    legacy = subparsers.add_parser("legacy-epoch-plots")
    legacy.add_argument("--metrics-path", type=Path, default=RESULTS_DIR)
    legacy.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "completeness":
        output = save_completeness(args.results_dir, args.output)
        print(f"Saved {args.output} with {len(output)} rows.")
    elif args.command == "plot-tuning":
        output = plot_tuning_combinations(
            dataset=args.dataset,
            model=args.model,
            metric=args.metric,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            scale_epochs=args.scale_epochs,
        )
        print(f"Saved {output}")
    elif args.command == "plot-dataset":
        output = plot_dataset_accuracy_loss(
            dataset=args.dataset,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            scale_epochs=args.scale_epochs,
            require_complete=not args.allow_incomplete,
        )
        print(f"Saved {output}")
    elif args.command == "summary-accuracy-loss":
        output = summarize_accuracy_loss(
            results_dir=args.results_dir,
            output_path=args.output,
            require_complete=not args.allow_incomplete,
        )
        print(f"Saved {args.output} with {len(output)} rows.")
    elif args.command == "summary-time":
        output = summarize_time_consumption(
            results_dir=args.results_dir,
            output_path=args.output,
            require_complete=not args.allow_incomplete,
        )
        print(f"Saved {args.output} with {len(output)} rows.")
    elif args.command == "stats":
        output = friedman_nemenyi_tests(
            metric=args.metric,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            require_complete=not args.allow_incomplete,
        )
        print(f"Saved statistical outputs: {', '.join(output)}")
    elif args.command == "dataset-metrics":
        output = calculate_table_metrics(
            cardinalities=args.cardinalities,
            output_dir=args.output_dir,
        )
        output_path = args.output_dir / "dataset_metrics.csv"
        output.to_csv(output_path, index=False)
        print(f"Saved {output_path} with {len(output)} rows.")
    elif args.command == "legacy-epoch-plots":
        generate_epoch_plots(args.metrics_path, args.output_dir)
        print(f"Saved epoch plots under {args.output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

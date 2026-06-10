from __future__ import annotations

import json
from pathlib import Path
import re

import click
import pandas as pd


MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tr", "rf"]

SEEDED_VERSION_PATTERN = re.compile(r"(?P<dataset>.+)_seed(?P<seed>-?\d+)$") # Pattern to extract dataset name and seed from Lightning result version strings, e.g. "dataset_seed42"

# Metrics to extract from Lightning logs for the final test epoch of each run. These will be included in the summary table.
TEST_METRICS = [
    "test_acc",
    "test_loss",
    "test_balanced_acc",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
]

# Metadata fields to extract from Lightning run metadata, if available. These include the number of trainable parameters, training time, and testing time for each run.
METADATA_FIELDS = [
    "trainable_parameters",
    "fit_seconds",
    "test_seconds",
]

def split_result_version(version: str) -> tuple[str, int | None]:
    # Split a Lightning result version into dataset name and seed
    match = SEEDED_VERSION_PATTERN.fullmatch(version)
    if match is None:
        return version, None

    return match.group("dataset"), int(match.group("seed"))

def read_final_test_metrics(results_dir: Path) -> pd.DataFrame:
    # Read final test metrics from Lightning CSV logs
    rows: list[dict[str, float | str]] = []

    # Iterate over all metrics.csv files in the results directory, extract the model name, dataset name, seed, and final test metrics for each run. 
    # If available, also extract metadata fields from the corresponding run_metadata.json file.
    for path in sorted(results_dir.glob("*/*/metrics.csv")):
        model, version = path.parts[-3], path.parts[-2]
        if model not in MODELS:
            continue
        dataset, seed = split_result_version(version)

        metrics = pd.read_csv(path)
        if "test_acc" not in metrics.columns:
            continue

        test_rows = metrics[metrics["test_acc"].notna()]
        if test_rows.empty:
            continue

        final_test = test_rows.iloc[-1]
        row = {
            "dataset": dataset,
            "seed": seed,
            "model": model,
        }
        for metric in TEST_METRICS:
            row[metric] = (
                float(final_test[metric])
                if metric in final_test and pd.notna(final_test[metric])
                else None
            )

        metadata_path = path.parent / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for field in METADATA_FIELDS:
                row[field] = metadata.get(field)

        rows.append(row)

    return pd.DataFrame(rows)

# Compute majority-class accuracy from processed test splits
def read_majority_baselines(
    processed_dir: Path,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute majority-class accuracy from processed test splits."""
    rows: list[dict[str, float | int | str]] = []

    for run in (
        runs[["dataset", "seed"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        seed_dir = (
            processed_dir / f"seed_{run.seed}"
            if pd.notna(run.seed)
            else processed_dir
        )
        test_path = seed_dir / f"{run.dataset}_tensor_test.csv"
        if not test_path.exists():
            test_path = seed_dir / f"{run.dataset}_baseline_test.csv"
        if not test_path.exists():
            continue

        test_df = pd.read_csv(test_path)
        targets = test_df["target"]

        rows.append(
            {
                "dataset": run.dataset,
                "seed": run.seed,
                "n_test": len(targets),
                "classes": targets.nunique(),
                "majority_acc": targets.value_counts(normalize=True).iloc[0],
            }
        )

    return pd.DataFrame(rows)

# Build a summary table by merging the final test metrics with the majority-class baselines for each dataset and seed. 
# The summary includes the test accuracy for each model, as well as the best model and its accuracy compared to the majority baseline
def build_summary(results_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Build one benchmark summary table."""
    metrics = read_final_test_metrics(results_dir)
    baselines = read_majority_baselines(processed_dir, metrics).set_index(
        ["dataset", "seed"]
    )

    accuracy_table = metrics.pivot(
        index=["dataset", "seed"],
        columns="model",
        values="test_acc",
    ).reindex(columns=MODELS)
    summary = baselines.join(accuracy_table)
    metric_suffixes = {
        "test_balanced_acc": "balanced_acc",
        "test_macro_precision": "macro_precision",
        "test_macro_recall": "macro_recall",
        "test_macro_f1": "macro_f1",
        "test_weighted_f1": "weighted_f1",
        "test_loss": "loss",
        "trainable_parameters": "parameters",
        "fit_seconds": "fit_seconds",
        "test_seconds": "test_seconds",
    }
    for metric, suffix in metric_suffixes.items():
        if metric not in metrics.columns:
            continue
        metric_table = metrics.pivot(
            index=["dataset", "seed"],
            columns="model",
            values=metric,
        ).reindex(columns=MODELS)
        summary = summary.join(metric_table.add_suffix(f"_{suffix}"))

    summary["best_model"] = accuracy_table.idxmax(axis=1)
    summary["best_acc"] = accuracy_table.max(axis=1)
    summary["best_minus_majority"] = (
        summary["best_acc"] - summary["majority_acc"]
    )

    for model in MODELS:
        summary[f"{model}_minus_majority"] = (
            summary[model] - summary["majority_acc"]
        )

    return summary.reset_index()

# Build an aggregate summary by grouping the seeded benchmark results by dataset and computing the mean and standard deviation for each metric across seeds. 
# The aggregate summary includes the average test accuracy for each model, as well as the average best model and its accuracy compared to the majority baseline, along with the number of runs for each dataset.
def build_aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seeded benchmark results by dataset."""
    metric_columns = [
        column
        for column in summary.columns
        if column in MODELS
        or any(
            column.endswith(f"_{suffix}")
            for suffix in [
                "balanced_acc",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "loss",
                "parameters",
                "fit_seconds",
                "test_seconds",
            ]
        )
        or column in {"majority_acc", "best_acc", "best_minus_majority"}
    ]

    grouped = summary.groupby("dataset", dropna=False)
    means = grouped[metric_columns].mean().add_suffix("_mean")
    stds = grouped[metric_columns].std(ddof=0).add_suffix("_std")
    counts = grouped.size().rename("n_runs")

    return pd.concat([counts, means, stds], axis=1).reset_index()


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results"),
    show_default=True,
)
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("src/data_pipeline/data/processed"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/benchmark_summary.csv"),
    show_default=True,
)
@click.option(
    "--aggregate-output",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/benchmark_summary_aggregate.csv"),
    show_default=True,
)
def main(
    results_dir: Path,
    processed_dir: Path,
    output: Path,
    aggregate_output: Path,
) -> None:
    """Create a benchmark summary CSV from result logs."""
    summary = build_summary(
        results_dir=results_dir,
        processed_dir=processed_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved {output} with {len(summary)} datasets.")

    if summary["seed"].notna().any():
        aggregate = build_aggregate_summary(summary)
        aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(aggregate_output, index=False)
        print(
            f"Saved {aggregate_output} with "
            f"{len(aggregate)} aggregated datasets."
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import re

import click
import pandas as pd


MODELS = ["lr", "mlp", "cpd", "tt", "tr"]
SEEDED_VERSION_PATTERN = re.compile(r"(?P<dataset>.+)_seed(?P<seed>-?\d+)$")


def split_result_version(version: str) -> tuple[str, int | None]:
    """Split a Lightning result version into dataset name and seed."""
    match = SEEDED_VERSION_PATTERN.fullmatch(version)
    if match is None:
        return version, None

    return match.group("dataset"), int(match.group("seed"))


def read_final_test_metrics(results_dir: Path) -> pd.DataFrame:
    """Read final test metrics from Lightning CSV logs."""
    rows: list[dict[str, float | str]] = []

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
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model,
                "test_acc": float(final_test["test_acc"]),
                "test_loss": float(final_test["test_loss"]),
                "test_balanced_acc": (
                    float(final_test["test_balanced_acc"])
                    if "test_balanced_acc" in final_test
                    and pd.notna(final_test["test_balanced_acc"])
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def read_majority_baselines(
    processed_dir: Path,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute majority-class accuracy from processed test splits."""
    rows: list[dict[str, float | int | str]] = []

    for run in runs[["dataset", "seed"]].drop_duplicates().itertuples(index=False):
        seed_dir = (
            processed_dir / f"seed_{run.seed}"
            if pd.notna(run.seed)
            else processed_dir
        )
        test_path = seed_dir / f"{run.dataset}_tensor_test.csv"
        if not test_path.exists():
            test_path = seed_dir / f"{run.dataset}_baseline_test.csv"

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
    balanced_accuracy_table = metrics.pivot(
        index=["dataset", "seed"],
        columns="model",
        values="test_balanced_acc",
    ).reindex(columns=MODELS)
    balanced_accuracy_table = balanced_accuracy_table.add_suffix(
        "_balanced_acc"
    )

    summary = baselines.join(accuracy_table).join(balanced_accuracy_table)
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


def build_aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seeded benchmark results by dataset."""
    metric_columns = [
        column
        for column in summary.columns
        if column in MODELS
        or column.endswith("_balanced_acc")
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
    default=Path("results"),
    show_default=True,
)
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("results/benchmark_summary.csv"),
    show_default=True,
)
@click.option(
    "--aggregate-output",
    type=click.Path(path_type=Path),
    default=Path("results/benchmark_summary_aggregate.csv"),
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

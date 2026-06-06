from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


MODELS = ["lr", "mlp", "cpd", "tt", "tr"]


def read_final_test_metrics(results_dir: Path) -> pd.DataFrame:
    """Read final test metrics from Lightning CSV logs."""
    rows: list[dict[str, float | str]] = []

    for path in sorted(results_dir.glob("*/*/metrics.csv")):
        model, dataset = path.parts[-3], path.parts[-2]
        if model not in MODELS:
            continue

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
                "model": model,
                "test_acc": float(final_test["test_acc"]),
                "test_loss": float(final_test["test_loss"]),
            }
        )

    return pd.DataFrame(rows)


def read_majority_baselines(processed_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """Compute majority-class accuracy from processed test splits."""
    rows: list[dict[str, float | int | str]] = []

    for dataset in datasets:
        test_path = processed_dir / f"{dataset}_tensor_test.csv"
        if not test_path.exists():
            test_path = processed_dir / f"{dataset}_baseline_test.csv"

        test_df = pd.read_csv(test_path)
        targets = test_df["target"]

        rows.append(
            {
                "dataset": dataset,
                "n_test": len(targets),
                "classes": targets.nunique(),
                "majority_acc": targets.value_counts(normalize=True).iloc[0],
            }
        )

    return pd.DataFrame(rows)


def build_summary(results_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Build one benchmark summary table."""
    metrics = read_final_test_metrics(results_dir)
    datasets = sorted(metrics["dataset"].unique())
    baselines = read_majority_baselines(processed_dir, datasets).set_index(
        "dataset"
    )

    accuracy_table = metrics.pivot(
        index="dataset",
        columns="model",
        values="test_acc",
    ).reindex(columns=MODELS)

    summary = baselines.join(accuracy_table)
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
def main(results_dir: Path, processed_dir: Path, output: Path) -> None:
    """Create a benchmark summary CSV from result logs."""
    summary = build_summary(
        results_dir=results_dir,
        processed_dir=processed_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved {output} with {len(summary)} datasets.")


if __name__ == "__main__":
    main()

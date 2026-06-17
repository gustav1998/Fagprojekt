from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_pipeline.dataset_configs import DATASET_CONFIGS


DEFAULT_PROCESSED_ROOT = Path("src/data_pipeline/data/processed")
DEFAULT_OUTPUT_DIR = Path("src/data_pipeline/data/reports")
SPLITS = ("tuning_train", "tuning_val", "fold_1_train", "fold_1_test")
REPRESENTATIONS = ("tensor", "baseline")


@dataclass(frozen=True)
class DatasetFiles:
    dataset: str
    representation: str
    metadata_path: Path
    split_paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize processed datasets before training."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Folder containing processed CSV files or seed folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Use processed-root/seed_<seed> when that folder exists.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to include. Repeat this flag to select several datasets.",
    )
    parser.add_argument(
        "--representation",
        choices=[*REPRESENTATIONS, "both"],
        default="tensor",
        help="Processed representation to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where CSV summaries and SVG plots are written.",
    )
    return parser.parse_args()


def resolve_processed_dir(processed_root: Path, seed: int | None) -> Path:
    if seed is None:
        return processed_root

    seed_dir = processed_root / f"seed_{seed}"
    if seed_dir.exists():
        return seed_dir

    return processed_root


def selected_representations(value: str) -> tuple[str, ...]:
    if value == "both":
        return REPRESENTATIONS
    return (value,)


def discover_dataset_files(
    processed_dir: Path,
    datasets: list[str] | None,
    representations: tuple[str, ...],
) -> list[DatasetFiles]:
    names = datasets if datasets else sorted(DATASET_CONFIGS)
    found: list[DatasetFiles] = []

    for dataset in names:
        for representation in representations:
            metadata_path = (
                processed_dir / f"{dataset}_{representation}_metadata.json"
            )
            split_paths = {
                split: processed_dir / f"{dataset}_{representation}_{split}.csv"
                for split in SPLITS
            }
            required_paths = [metadata_path, *split_paths.values()]
            if all(path.exists() for path in required_paths):
                found.append(
                    DatasetFiles(
                        dataset=dataset,
                        representation=representation,
                        metadata_path=metadata_path,
                        split_paths=split_paths,
                    )
                )
                continue

            if datasets:
                missing = ", ".join(
                    str(path) for path in required_paths if not path.exists()
                )
                raise FileNotFoundError(
                    f"Missing processed files for {dataset}/{representation}: {missing}"
                )

    if not found:
        raise FileNotFoundError(
            f"No processed datasets found in {processed_dir}. "
            "Run src.data_pipeline.make_dataset or src.training.run_experiments first."
        )

    return found


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def label_lookup(metadata: dict[str, Any]) -> dict[int, str]:
    mapping = metadata.get("target_mapping", {})
    return {int(value): str(key) for key, value in mapping.items()}


def summarize_files(
    files: DatasetFiles,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = load_json(files.metadata_path)
    labels = label_lookup(metadata)
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for split, path in files.split_paths.items():
        df = pd.read_csv(path)
        target = df["target"]
        counts = target.value_counts().sort_index()
        majority_label = int(counts.idxmax())
        majority_count = int(counts.max())

        summary_rows.append(
            {
                "dataset": files.dataset,
                "representation": files.representation,
                "split": split,
                "rows": int(len(df)),
                "features": int(len(df.columns) - 1),
                "classes": int(target.nunique()),
                "majority_class_id": majority_label,
                "majority_class": labels.get(majority_label, str(majority_label)),
                "majority_count": majority_count,
                "majority_share": majority_count / len(df) if len(df) else 0.0,
                "missing_cells": int(df.isna().sum().sum()),
            }
        )

        for class_id, count in counts.items():
            class_id = int(class_id)
            class_rows.append(
                {
                    "dataset": files.dataset,
                    "representation": files.representation,
                    "split": split,
                    "class_id": class_id,
                    "class_label": labels.get(class_id, str(class_id)),
                    "count": int(count),
                    "share": int(count) / len(df) if len(df) else 0.0,
                }
            )

    categorical = set(metadata.get("categorical_columns", []))
    numerical = set(metadata.get("numerical_columns", []))
    cardinalities = metadata.get("cardinalities", [])
    for index, feature in enumerate(metadata.get("feature_names", [])):
        if feature in categorical:
            feature_type = "categorical"
        elif feature in numerical:
            feature_type = "numerical"
        else:
            feature_type = "encoded"

        feature_rows.append(
            {
                "dataset": files.dataset,
                "representation": files.representation,
                "feature": feature,
                "feature_type": feature_type,
                "cardinality": (
                    int(cardinalities[index]) if index < len(cardinalities) else None
                ),
            }
        )

    return summary_rows, class_rows, feature_rows


def write_csv_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    feature_cardinalities: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    feature_cardinalities.to_csv(output_dir / "feature_cardinalities.csv", index=False)


def format_number(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def write_horizontal_bar_svg(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    value_suffix: str = "",
    max_rows: int = 32,
) -> None:
    rows = sorted(rows, key=lambda row: row[value_key], reverse=True)[:max_rows]
    width = 980
    row_height = 28
    top = 58
    left = 210
    right = 70
    height = top + len(rows) * row_height + 36
    chart_width = width - left - right
    max_value = max((float(row[value_key]) for row in rows), default=1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#172033}",
        ".title{font-size:22px;font-weight:700}.label{font-size:12px}.value{font-size:12px;fill:#39475f}",
        ".axis{stroke:#d7deea;stroke-width:1}.bar{fill:#356f8c}.bar.alt{fill:#d18454}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="34">{html.escape(title)}</text>',
        f'<line class="axis" x1="{left}" y1="{top - 12}" x2="{left + chart_width}" y2="{top - 12}"/>',
    ]

    for index, row in enumerate(rows):
        y = top + index * row_height
        value = float(row[value_key])
        bar_width = 0 if max_value == 0 else chart_width * value / max_value
        css_class = "bar alt" if index % 2 else "bar"
        label = str(row[label_key])
        formatted = format_number(value)
        parts.extend(
            [
                f'<text class="label" x="24" y="{y + 16}">{html.escape(label[:32])}</text>',
                f'<rect class="{css_class}" x="{left}" y="{y}" width="{bar_width:.1f}" height="18" rx="2"/>',
                f'<text class="value" x="{left + bar_width + 8:.1f}" y="{y + 14}">{formatted}{html.escape(value_suffix)}</text>',
            ]
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_dataset_size_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    totals = (
        summary.groupby(["dataset", "representation"], as_index=False)["rows"]
        .sum()
        .assign(label=lambda df: df["dataset"] + " (" + df["representation"] + ")")
    )
    write_horizontal_bar_svg(
        output_dir / "dataset_sizes.svg",
        "Rows per processed dataset",
        totals.to_dict("records"),
        label_key="label",
        value_key="rows",
    )


def write_majority_share_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    test_rows = summary[summary["split"] == "fold_1_test"].copy()
    test_rows["majority_percent"] = test_rows["majority_share"] * 100
    test_rows["label"] = (
        test_rows["dataset"]
        + " ("
        + test_rows["representation"]
        + ", "
        + test_rows["classes"].astype(str)
        + " classes)"
    )
    write_horizontal_bar_svg(
        output_dir / "majority_class_share.svg",
        "Test-set majority class share",
        test_rows.to_dict("records"),
        label_key="label",
        value_key="majority_percent",
        value_suffix="%",
    )


def write_feature_cardinality_plot(
    feature_cardinalities: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows = feature_cardinalities.dropna(subset=["cardinality"]).copy()
    rows["label"] = (
        rows["dataset"]
        + " / "
        + rows["feature"]
        + " ("
        + rows["representation"]
        + ")"
    )
    write_horizontal_bar_svg(
        output_dir / "largest_feature_cardinalities.svg",
        "Largest feature cardinalities",
        rows.to_dict("records"),
        label_key="label",
        value_key="cardinality",
        max_rows=40,
    )


def write_html_report(output_dir: Path, summary: pd.DataFrame) -> None:
    totals = summary.groupby("representation")["dataset"].nunique().to_dict()
    dataset_count = summary["dataset"].nunique()
    split_count = len(summary)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dataset Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #39475f; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metric {{ border: 1px solid #d7deea; border-radius: 6px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 24px; margin-bottom: 4px; }}
    iframe {{ width: 100%; border: 1px solid #d7deea; border-radius: 6px; margin: 16px 0 28px; }}
    code {{ background: #f3f5f8; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Processed Dataset Report</h1>
  <p>Generated from processed CSV files. Use this before training to check size, class imbalance, and feature cardinalities.</p>
  <div class="grid">
    <div class="metric"><strong>{dataset_count}</strong>datasets</div>
    <div class="metric"><strong>{split_count}</strong>dataset/representation/split rows</div>
    <div class="metric"><strong>{totals.get("tensor", 0)}</strong>tensor datasets</div>
    <div class="metric"><strong>{totals.get("baseline", 0)}</strong>baseline datasets</div>
  </div>
  <h2>Dataset Size</h2>
  <iframe src="dataset_sizes.svg" height="720"></iframe>
  <h2>Class Imbalance</h2>
  <iframe src="majority_class_share.svg" height="720"></iframe>
  <h2>Feature Cardinality</h2>
  <iframe src="largest_feature_cardinalities.svg" height="900"></iframe>
  <p>CSV outputs: <code>dataset_summary.csv</code>, <code>class_distribution.csv</code>, and <code>feature_cardinalities.csv</code>.</p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_report(
    processed_root: Path,
    seed: int | None,
    datasets: list[str] | None,
    representation: str,
    output_dir: Path,
) -> Path:
    processed_dir = resolve_processed_dir(processed_root, seed)
    files = discover_dataset_files(
        processed_dir,
        datasets,
        selected_representations(representation),
    )

    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for dataset_files in files:
        dataset_summary, dataset_classes, dataset_features = summarize_files(
            dataset_files
        )
        summary_rows.extend(dataset_summary)
        class_rows.extend(dataset_classes)
        feature_rows.extend(dataset_features)

    summary = pd.DataFrame(summary_rows)
    class_distribution = pd.DataFrame(class_rows)
    feature_cardinalities = pd.DataFrame(feature_rows)

    write_csv_outputs(output_dir, summary, class_distribution, feature_cardinalities)
    write_dataset_size_plot(summary, output_dir)
    write_majority_share_plot(summary, output_dir)
    write_feature_cardinality_plot(feature_cardinalities, output_dir)
    write_html_report(output_dir, summary)

    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = build_report(
        processed_root=args.processed_root,
        seed=args.seed,
        datasets=args.datasets,
        representation=args.representation,
        output_dir=args.output_dir,
    )
    print(f"Wrote dataset report to {output_dir}")


if __name__ == "__main__":
    main()

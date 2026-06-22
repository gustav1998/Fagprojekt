"""
Compute true feature cardinalities from preprocessed metadata.

Categorical cardinalities exclude the 2 reserved tokens (__MISSING__, __UNKNOWN__)
that are always added during preprocessing. Numerical cardinalities are the number
of bins and are already correct.

Output: src/data_pipeline/data/reports/true_cardinalities.csv
"""

import json
from pathlib import Path

import click
import pandas as pd


def compute_cardinalities(processed_dir: Path) -> pd.DataFrame:
    rows = []

    for meta_path in sorted(processed_dir.glob("*_tensor_metadata.json")):
        dataset = meta_path.stem.replace("_tensor_metadata", "")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        for col, mapping in meta.get("feature_mappings", {}).items():
            true_cardinality = len(mapping) - 2  # subtract __MISSING__ and __UNKNOWN__
            rows.append({
                "dataset": dataset,
                "feature": col,
                "feature_type": "categorical",
                "cardinality": true_cardinality,
            })

        num_bin_metadata = meta.get("numerical_bin_metadata", {})
        for col, bin_info in num_bin_metadata.items():
            rows.append({
                "dataset": dataset,
                "feature": col,
                "feature_type": "numerical",
                "cardinality": int(bin_info["n_bins"]),
            })

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("src/data_pipeline/data/processed"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("src/data_pipeline/data/reports/true_cardinalities.csv"),
    show_default=True,
)
def main(processed_dir: Path, output: Path) -> None:
    df = compute_cardinalities(processed_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {output} with {len(df)} features across {df['dataset'].nunique()} datasets.")

    print("\nPer-dataset summary (max and mean cardinality across features):")
    summary = df.groupby("dataset")["cardinality"].agg(
        n_features="count",
        max_cardinality="max",
        mean_cardinality="mean",
    ).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
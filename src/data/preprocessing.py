from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_dataset(config: dict[str, Any], raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / config["file_name"]

    df = pd.read_csv(
        file_path,
        sep=config.get("sep", ","),
        engine=config.get("engine"),
        header=config.get("header", "infer"),
        names=config.get("column_names"),
        na_values=config.get("missing_tokens", ["?"]),
        keep_default_na=True,
        skipinitialspace=True,
    )

    drop_columns = config.get("drop_columns", [])
    if drop_columns:
        df = df.drop(columns=drop_columns)

    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str] | str | None,
    numerical_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
    if categorical_columns == "all_except_target":
        categorical = [col for col in df.columns if col != target_column]
    else:
        categorical = categorical_columns or []

    numerical = numerical_columns or []

    overlap = set(categorical) & set(numerical)
    if overlap:
        raise ValueError(f"Columns cannot be both categorical and numerical: {sorted(overlap)}")

    unknown = (set(categorical) | set(numerical) | {target_column}) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown columns found in config: {sorted(unknown)}")

    return categorical, numerical


def fill_missing_values(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    df = df.copy()

    for col in [target_column] + categorical_columns:
        df[col] = df[col].fillna(missing_token)

    for col in numerical_columns:
        median = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
        if pd.isna(median):
            median = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    return df.reset_index(drop=True)


def encode_column(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    categories = sorted(series.astype(str).unique())
    mapping = {value: idx for idx, value in enumerate(categories)}
    encoded = series.astype(str).map(mapping)
    return encoded, mapping


def encode_categorical_dataframe(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], list[int]]:
    encoded_df = pd.DataFrame(index=df.index)
    mappings: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []

    for col in categorical_columns:
        encoded_col, mapping = encode_column(df[col])
        encoded_df[col] = encoded_col.astype(int)
        mappings[col] = mapping
        cardinalities.append(len(mapping))

    return encoded_df, mappings, cardinalities


def discretize_numerical_dataframe(
    df: pd.DataFrame,
    numerical_columns: list[str],
    n_bins: int = 10,
    strategy: str = "quantile",
) -> tuple[pd.DataFrame, dict[str, Any], list[int]]:
    discretized_df = pd.DataFrame(index=df.index)
    bin_metadata: dict[str, Any] = {}
    cardinalities: list[int] = []

    for col in numerical_columns:
        series = pd.to_numeric(df[col], errors="coerce")

        if strategy == "quantile":
            binned, edges = pd.qcut(
                series,
                q=min(n_bins, series.nunique()),
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        elif strategy == "uniform":
            binned, edges = pd.cut(
                series,
                bins=min(n_bins, series.nunique()),
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")

        binned = binned.astype(int)
        discretized_df[col] = binned
        cardinality = int(binned.max()) + 1
        cardinalities.append(cardinality)
        bin_metadata[col] = {
            "strategy": strategy,
            "edges": [float(x) for x in edges],
            "n_bins": cardinality,
        }

    return discretized_df, bin_metadata, cardinalities


def process_dataset(
    df: pd.DataFrame,
    config: dict[str, Any],
    representation: str = "baseline",
    n_bins: int | None = None,
    binning_strategy: str = "quantile",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if representation not in {"baseline", "tensor"}:
        raise ValueError(f"Unsupported representation: {representation}")

    target_column = config["target_column"]
    categorical_columns, numerical_columns = infer_feature_columns(
        df=df,
        target_column=target_column,
        categorical_columns=config.get("categorical_columns"),
        numerical_columns=config.get("numerical_columns"),
    )

    df = fill_missing_values(df, target_column, categorical_columns, numerical_columns)

    y_encoded, target_mapping = encode_column(df[target_column])

    X_cat_encoded, feature_mappings, cat_cardinalities = encode_categorical_dataframe(
        df=df,
        categorical_columns=categorical_columns,
    )

    n_bins = n_bins or config.get("default_n_bins", 10)

    numerical_bin_metadata: dict[str, Any] = {}
    numerical_cardinalities: list[int] = []

    if representation == "baseline":
        X_num = df[numerical_columns].copy() if numerical_columns else pd.DataFrame(index=df.index)
        processed_df = pd.concat([X_cat_encoded, X_num], axis=1)
        cardinalities = cat_cardinalities

    else:  # tensor
        if numerical_columns:
            X_num_disc, numerical_bin_metadata, numerical_cardinalities = discretize_numerical_dataframe(
                df=df,
                numerical_columns=numerical_columns,
                n_bins=n_bins,
                strategy=binning_strategy,
            )
        else:
            X_num_disc = pd.DataFrame(index=df.index)

        processed_df = pd.concat([X_cat_encoded, X_num_disc], axis=1)
        cardinalities = cat_cardinalities + numerical_cardinalities

    processed_df["target"] = y_encoded.astype(int)

    metadata = {
        "target_column": target_column,
        "feature_names": categorical_columns + numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "representation": representation,
        "cardinalities": cardinalities,
        "categorical_cardinalities": cat_cardinalities,
        "numerical_cardinalities": numerical_cardinalities,
        "target_mapping": target_mapping,
        "feature_mappings": feature_mappings,
        "numerical_bin_metadata": numerical_bin_metadata,
        "n_features": len(categorical_columns) + len(numerical_columns),
    }

    return processed_df, metadata


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "target",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train_size + val_size + test_size must sum to 1.0, got {total}")

    def split_with_optional_stratify(
        data: pd.DataFrame,
        test_fraction: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        y = data[target_column]
        class_counts = y.value_counts()
        can_stratify = y.nunique() > 1 and class_counts.min() >= 2

        if can_stratify:
            try:
                return train_test_split(
                    data,
                    test_size=test_fraction,
                    stratify=y,
                    random_state=random_state,
                )
            except ValueError:
                pass

        return train_test_split(
            data,
            test_size=test_fraction,
            stratify=None,
            random_state=random_state,
        )

    train_df, temp_df = split_with_optional_stratify(df, test_fraction=(1.0 - train_size))
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = split_with_optional_stratify(temp_df, test_fraction=relative_test_size)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    representation: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / f"{dataset_name}_{representation}_train.csv",
        "val": output_dir / f"{dataset_name}_{representation}_val.csv",
        "test": output_dir / f"{dataset_name}_{representation}_test.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return paths


def save_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    representation: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{dataset_name}_{representation}_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path
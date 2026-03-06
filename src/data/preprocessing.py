from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_dataset(config: dict[str, Any], raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / config["file_name"]

    df = pd.read_csv(
        file_path,
        sep=config.get("sep", ","),
        header=config.get("header", "infer"),
        names=config.get("column_names"),
        na_values=config.get("missing_tokens", ["?"]),
        keep_default_na=True,
    )
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


def drop_missing_rows(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    relevant_columns = [target_column] + categorical_columns + numerical_columns
    return df.dropna(subset=relevant_columns).reset_index(drop=True)


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


def process_dataset(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_column = config["target_column"]
    categorical_columns, numerical_columns = infer_feature_columns(
        df=df,
        target_column=target_column,
        categorical_columns=config.get("categorical_columns"),
        numerical_columns=config.get("numerical_columns"),
    )

    df = drop_missing_rows(df, target_column, categorical_columns, numerical_columns)

    y_encoded, target_mapping = encode_column(df[target_column])

    X_cat_encoded, feature_mappings, cardinalities = encode_categorical_dataframe(
        df=df,
        categorical_columns=categorical_columns,
    )

    X_num = df[numerical_columns].copy() if numerical_columns else pd.DataFrame(index=df.index)

    processed_df = pd.concat([X_cat_encoded, X_num], axis=1)
    processed_df["target"] = y_encoded.astype(int)

    metadata = {
        "target_column": target_column,
        "feature_names": categorical_columns + numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "cardinalities": cardinalities,
        "target_mapping": target_mapping,
        "feature_mappings": feature_mappings,
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

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[target_column],
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[target_column],
        random_state=random_state,
    )

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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / f"{dataset_name}_train.csv",
        "val": output_dir / f"{dataset_name}_val.csv",
        "test": output_dir / f"{dataset_name}_test.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return paths


def save_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
    dataset_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{dataset_name}_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path
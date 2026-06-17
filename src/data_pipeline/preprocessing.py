from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_dataset(config: dict[str, Any], raw_dir: Path) -> pd.DataFrame:
    """Load a raw dataset from its configuration."""
    file_name = config.get("file_name") or config.get("train_file_name")
    if file_name is None:
        raise ValueError(
            "Dataset config must define file_name or train_file_name"
        )

    file_path = raw_dir / file_name

    read_csv_options = {
        "sep": config.get("sep", ","),
        "engine": config.get("engine"),
        "header": config.get("header", "infer"),
        "names": config.get("column_names"),
        "na_values": config.get("missing_tokens", ["?"]),
        "keep_default_na": True,
        "skipinitialspace": True,
    }
    for option in [
        "skiprows",
        "compression",
        "comment",
        "encoding",
        "low_memory",
    ]:
        if option in config:
            read_csv_options[option] = config[option]

    # load dataset with configured options
    df = pd.read_csv(
        file_path,
        **read_csv_options,
    )

    # drop any columns explicitly listed for removal in the config
    drop_columns = config.get("drop_columns", [])
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    return df

# load raw dataset splits with optional official test set 
def load_raw_dataset_splits(
    config: dict[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:

    if "train_file_name" not in config or "test_file_name" not in config:
        return load_raw_dataset(config=config, raw_dir=raw_dir), None

    train_config = {**config, "file_name": config["train_file_name"]}
    test_config = {**config, "file_name": config["test_file_name"]}
    return (
        load_raw_dataset(config=train_config, raw_dir=raw_dir),
        load_raw_dataset(config=test_config, raw_dir=raw_dir),
    )

# infer feature columns based on config, ensuring no overlap and all columns exist in the dataframe
def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str] | str | None,
    numerical_columns: list[str] | str | None,
) -> tuple[list[str], list[str]]:
    """Resolve categorical and numerical feature columns."""
  
    if categorical_columns == "all_except_target":
        categorical = [col for col in df.columns if col != target_column]
    else:
        categorical = categorical_columns or []

    if numerical_columns == "all_except_target":
        numerical = [col for col in df.columns if col != target_column]
    else:
        numerical = numerical_columns or []

    # check for column overlap
    overlap = set(categorical) & set(numerical)
    if overlap:
        raise ValueError(
            "Columns cannot be both categorical and numerical: "
            f"{sorted(overlap)}"
        )

    # check that all specified columns in categorical, numerical, and target are present in the dataframe
    unknown = (
        set(categorical) | set(numerical) | {target_column}
    ) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown columns found in config: {sorted(unknown)}")

    return categorical, numerical

# handles missing values by filling categorical columns with a missing token and numerical columns with the median
def fill_missing_values(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    """Fill missing target, categorical, and numerical values."""
    df = df.copy()

    for col in [target_column] + categorical_columns:
        df[col] = df[col].fillna(missing_token)

    for col in numerical_columns:
        median = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
        if pd.isna(median):
            median = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    return df.reset_index(drop=True)

# handles outliers and drops rows with missing or rare targets based on config settings
def clean_targets(
    df: pd.DataFrame,
    target_column: str,
    min_target_count: int | None = None,
) -> pd.DataFrame:
    """Drop rows without usable targets and optionally remove rare classes."""
    df = df.dropna(subset=[target_column]).copy()

    if min_target_count is not None and min_target_count > 1:
        counts = df[target_column].value_counts()
        keep = counts[counts >= min_target_count].index
        df = df[df[target_column].isin(keep)].copy()

    return df.reset_index(drop=True)

# encodes categorical columns and target column
def build_mapping(
    series: pd.Series,
    extra_values: list[str] | None = None,
) -> dict[str, int]:
    """Create a stable string-to-integer mapping."""
    values = set(series.astype(str).unique())
    values.update(extra_values or [])
    return {value: idx for idx, value in enumerate(sorted(values))}

# applies the mapping
def apply_mapping(
    series: pd.Series,
    mapping: dict[str, int],
    unknown_token: str | None = None,
) -> pd.Series:
    """Map values with an optional fallback for unseen categories."""
    values = series.astype(str)

    if unknown_token is not None:
        values = values.where(values.isin(mapping), unknown_token)
    encoded = values.map(mapping)
    if encoded.isna().any():
        unknown_values = sorted(values[encoded.isna()].unique())
        raise ValueError(
            f"Values not found in fitted mapping: {unknown_values}"
        )
    return encoded.astype(int)

 # handles the fitting of numerical discretization bins for all the different datasets based on the config settings, using either quantile or uniform binning strategies
def fit_numerical_bins(
    df: pd.DataFrame,
    numerical_columns: list[str],
    n_bins: int,
    strategy: str,
) -> tuple[dict[str, Any], list[int]]:
    """Fit numerical discretization bins on training data."""
    bin_metadata: dict[str, Any] = {}
    cardinalities: list[int] = []

    for col in numerical_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        unique_count = int(series.nunique())

        if unique_count <= 1: # if there is only one unique value, we can just create a single bin for it
            edges = [float(series.min()), float(series.max())] #
            cardinality = 1
        elif strategy == "quantile":
            _, edges_array = pd.qcut(
                series,
                q=min(n_bins, unique_count),
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            edges = [float(x) for x in edges_array]
            cardinality = max(1, unique_count) # cardinality is number of bins, which is one less than number of edges
        elif strategy == "uniform":
            _, edges_array = pd.cut(
                series,
                bins=min(n_bins, unique_count),
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            edges = [float(x) for x in edges_array]
            cardinality = max(1, unique_count)
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")

        bin_metadata[col] = {
            "strategy": strategy,
            "edges": edges,
            "n_bins": cardinality,
        }
        cardinalities.append(cardinality)

    return bin_metadata, cardinalities

# applies the fitted numerical discretization bins to a dataframe
def apply_numerical_bins(
    df: pd.DataFrame,
    numerical_columns: list[str],
    numerical_bin_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Apply fitted numerical discretization bins."""
    discretized_columns: dict[str, pd.Series | int] = {}

    for col in numerical_columns:
        metadata = numerical_bin_metadata[col]
        cardinality = int(metadata["n_bins"])

        if cardinality <= 1:
            discretized_columns[col] = 0
            continue

        edges = [float(x) for x in metadata["edges"]]
        cut_edges = [-np.inf, *edges[1:-1], np.inf] # ensure all values are binned even if they fall outside the original edges
        binned = pd.cut(
            pd.to_numeric(df[col], errors="coerce"),
            bins=cut_edges,
            labels=False,
            include_lowest=True,
        )
        discretized_columns[col] = binned.astype(int)

    return pd.DataFrame(discretized_columns, index=df.index)

# main function to fit the preprocessor on the training data and then transform a raw dataframe into the processed representation using the fitted metadata
def fit_preprocessor(
    train_df: pd.DataFrame,
    config: dict[str, Any],
    representation: str = "baseline",
    n_bins: int | None = None,
    binning_strategy: str = "quantile",
    missing_token: str = "__MISSING__",
    unknown_token: str = "__UNKNOWN__",
) -> dict[str, Any]:
    """Fit preprocessing metadata on training data only."""
    if representation not in {"baseline", "tensor"}:
        raise ValueError(f"Unsupported representation: {representation}")

    target_column = config["target_column"] # sets the target column from the config
    categorical_columns, numerical_columns = infer_feature_columns( # infers the categorical and numerical feature columns based on the config settings and the dataframe columns
        df=train_df,
        target_column=target_column,
        categorical_columns=config.get("categorical_columns"),
        numerical_columns=config.get("numerical_columns"),
    )

    # fills missing values in the training data 
    train_df = fill_missing_values(
        train_df,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        missing_token=missing_token,
    )

    target_mapping = build_mapping(train_df[target_column])

    feature_mappings: dict[str, dict[str, int]] = {}
    cat_cardinalities: list[int] = []
    for col in categorical_columns:
        mapping = build_mapping(
            train_df[col],
            extra_values=[missing_token, unknown_token],
        )
        feature_mappings[col] = mapping
        cat_cardinalities.append(len(mapping))

    # handles missing values in numerical columns by filling with the median and also computes the median fill values for each numerical column to be included in the metadata for use when transforming validation and test splits
    numerical_fill_values: dict[str, float] = {}
    for col in numerical_columns:
        median = pd.to_numeric(
            train_df[col],
            errors="coerce",
        ).median(skipna=True)
        if pd.isna(median):
            median = 0.0
        numerical_fill_values[col] = float(median)
        train_df[col] = pd.to_numeric(
            train_df[col],
            errors="coerce",
        ).fillna(median)

    n_bins = n_bins or config.get("default_n_bins", 10)
    numerical_bin_metadata: dict[str, Any] = {}
    numerical_cardinalities: list[int] = []

    
    if representation == "tensor" and numerical_columns:
        numerical_bin_metadata, numerical_cardinalities = fit_numerical_bins(
            df=train_df,
            numerical_columns=numerical_columns,
            n_bins=n_bins,
            strategy=binning_strategy,
        )

    cardinalities = cat_cardinalities + numerical_cardinalities

    # returns a dictionary containing all the fitted metadata needed to transform raw dataframes into the processed representation
    return {
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
        "numerical_fill_values": numerical_fill_values,
        "numerical_bin_metadata": numerical_bin_metadata,
        "missing_token": missing_token,
        "unknown_token": unknown_token,
        "n_features": len(categorical_columns) + len(numerical_columns),
    }

# transforms a raw dataframe into the processed representation using the fitted metadata
def transform_dataset(
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:

    target_column = metadata["target_column"]
    categorical_columns = metadata["categorical_columns"]
    numerical_columns = metadata["numerical_columns"]
    missing_token = metadata.get("missing_token", "__MISSING__")
    unknown_token = metadata.get("unknown_token", "__UNKNOWN__")

    df = fill_missing_values(
        df,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=[],
        missing_token=missing_token,
    )

    processed_df = pd.DataFrame(index=df.index)
    for col in categorical_columns:
        processed_df[col] = apply_mapping(
            df[col],
            metadata["feature_mappings"][col],
            unknown_token=unknown_token,
        )

    for col in numerical_columns:
        fill_value = metadata.get("numerical_fill_values", {}).get(col, 0.0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)

    if metadata["representation"] == "baseline":
        if numerical_columns:
            processed_df = pd.concat(
                [processed_df, df[numerical_columns].copy()],
                axis=1,
            )
    else:
        if numerical_columns:
            processed_df = pd.concat(
                [
                    processed_df,
                    apply_numerical_bins(
                        df=df,
                        numerical_columns=numerical_columns,
                        numerical_bin_metadata=metadata[
                            "numerical_bin_metadata"
                        ],
                    ),
                ],
                axis=1,
            )

    processed_df["target"] = apply_mapping(
        df[target_column],
        metadata["target_mapping"],
    )
    return processed_df.reset_index(drop=True)

 # processes a raw dataset into one model input representation by fitting the preprocessor on the training data and then transforming the entire raw dataframe using the fitted metadata, returning both the processed dataframe and the metadata
def process_dataset(
    df: pd.DataFrame,
    config: dict[str, Any],
    representation: str = "baseline",
    n_bins: int | None = None,
    binning_strategy: str = "quantile",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process a raw dataset into one model input representation."""
    df = clean_targets(
        df,
        target_column=config["target_column"],
        min_target_count=config.get("min_target_count"),
    )
    metadata = fit_preprocessor(
        train_df=df,
        config=config,
        representation=representation,
        n_bins=n_bins,
        binning_strategy=binning_strategy,
    )
    return transform_dataset(df, metadata), metadata

# defines the dataset splitting function which handles both the case where there is an official test set and the case where there is not, ensuring that the splits are stratified if possible
def split_dataset(
    df: pd.DataFrame,
    target_column: str = "target",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train, validation, and test sets."""
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "train_size + val_size + test_size must sum to 1.0, "
            f"got {total}"
        )

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

    train_df, temp_df = split_with_optional_stratify(
        df,
        test_fraction=(1.0 - train_size),
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = split_with_optional_stratify(
        temp_df,
        test_fraction=relative_test_size,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

# defines the train-val splitting function for the case where there is an official test set, ensuring that the split is stratified if possible
def split_train_val(
    df: pd.DataFrame,
    target_column: str,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an official training set into train and validation splits."""
    y = df[target_column]
    class_counts = y.value_counts()
    can_stratify = y.nunique() > 1 and class_counts.min() >= 2

    if can_stratify:
        try:
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                stratify=y,
                random_state=random_state,
            )
            return (
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
            )
        except ValueError:
            pass

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=None,
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )

# saves the processed train, validation, and test splits as CSV files in the specified output directory
def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    representation: str,
) -> dict[str, Path]:
    """Save train, validation, and test splits as CSV files."""
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

# saves the fitted preprocessing metadata as a JSON file in the specified output directory
def save_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    representation: str,
) -> Path:
    """Save processing metadata as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = (
        output_dir / f"{dataset_name}_{representation}_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

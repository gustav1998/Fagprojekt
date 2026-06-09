from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

# loads the processed CSV splits and metadata for a given dataset and representation from the specified processed directory
def load_processed_dataset(
    dataset_name: str,
    representation: str,
    processed_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    """Load processed CSV splits and metadata."""
    processed_dir = Path(processed_dir)

    train_path = processed_dir / f"{dataset_name}_{representation}_train.csv"
    val_path = processed_dir / f"{dataset_name}_{representation}_val.csv"
    test_path = processed_dir / f"{dataset_name}_{representation}_test.csv"
    metadata_path = (
        processed_dir / f"{dataset_name}_{representation}_metadata.json"
    )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "metadata": metadata,
    }

# transforms the feature columns of the given DataFrame into a tensor and the target column into a tensor of labels, placing them on the specified device (CPU or GPU)
def dataframe_to_tensors(
    df: pd.DataFrame,
    target_column: str = "target",
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a DataFrame into feature and target tensors."""
    X = df.drop(columns=[target_column]).to_numpy(dtype="float32")
    y = df[target_column].to_numpy(dtype="int64")

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor

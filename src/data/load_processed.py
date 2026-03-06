from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def load_processed_dataset(
    dataset_name: str,
    processed_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)

    train_path = processed_dir / f"{dataset_name}_train.csv"
    val_path = processed_dir / f"{dataset_name}_val.csv"
    test_path = processed_dir / f"{dataset_name}_test.csv"
    metadata_path = processed_dir / f"{dataset_name}_metadata.json"

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


def dataframe_to_tensors(
    df: pd.DataFrame,
    target_column: str = "target",
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    X = df.drop(columns=[target_column]).to_numpy(dtype="int64")
    y = df[target_column].to_numpy(dtype="int64")

    X_tensor = torch.tensor(X, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor
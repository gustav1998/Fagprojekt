# %%
from pathlib import Path

import json
import pandas as pd
import torch

# %% Loads

def load_processed_dataset(dataset_name, representation, mode, processed_dir="src/data_pipeline/data/processed", fold=None): 
    processed_dir = Path(processed_dir) # converts the string path to a Path object for use later

    if mode == "tuning":
        train_path = processed_dir / f"{dataset_name}_{representation}_tuning_train.csv" # this and the next line builds the file paths for the CSV's
        val_path = processed_dir / f"{dataset_name}_{representation}_tuning_val.csv"
    else: 
        if fold is None: # fail safe if no fold number was given
            raise ValueError("fold must be specified when mode is 'fold'")
        
        train_path = processed_dir / f"{dataset_name}_{representation}_fold_{fold}_train.csv" # this and the next builds the file paths for a specific given fold (since we now checked whether one was specified)
        val_path = processed_dir / f"{dataset_name}_{representation}_fold_{fold}_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    metadata_path = processed_dir / f"{dataset_name}_{representation}_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "metadata": metadata,
    }

# %% Conversion (converts neural models columns to numpy array of floats and then to PyTorch tensors)

def dataframe_to_tensors(df, target_column="target", device="cpu"):
    X = df.drop(columns=[target_column]).to_numpy(dtype="float32") # drops target, converts to numpy array of float
    y = df[target_column].to_numpy(dtype="int64") # converts target column to numpy array of floats

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device) # this and the next converts both numpy arrays to PyTorch tensors and places them on the specified device
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor

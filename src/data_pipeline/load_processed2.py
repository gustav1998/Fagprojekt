# %%
from pathlib import Path

import json
import pandas as pd
import torch

# %%

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
        


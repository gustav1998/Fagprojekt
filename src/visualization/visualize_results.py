from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn as sk
import json
import sys
from itertools import combinations
from math import prod
from pathlib import Path

def calculate_metrics(results: pd.DataFrame, cardinalities_path: str | None = None) -> pd.DataFrame:
    '''
    Calculate the relevant metrics aggregated per dataset. An example of the input is given below:

    dataset,representation,split,rows,features,classes,majority_class_id,majority_class,majority_count,majority_share,missing_cells
    aps_failure,tensor,train,51000,170,2,...
    aps_failure,tensor,val,9000,170,2,...
    aps_failure,tensor,test,16000,170,2,...

    For each dataset, we calculate and return (aggregating across splits, using first representation):
    * the total number of rows (summed across splits)
    * the number of features (constant per dataset, +1 for classification feature)
    * the tensor size Ω of the dataset (product of all feature cardinalities * classes)
    * the sparsity (missing_cells / tensor_size)
    * the number of non-zero entries (actual cells with data)
    * the number of classes (constant per dataset)
    '''
    # Load cardinalities if path is provided
    cardinalities_df = None
    if cardinalities_path and Path(cardinalities_path).exists():
        cardinalities_df = pd.read_csv(cardinalities_path)
    
    results_list = []

    # Group by dataset, aggregate across splits, using first representation
    for name in results['dataset'].unique():
        # Get all rows for this dataset and take first representation
        dataset_subset = results[results['dataset'] == name]
        first_rep = dataset_subset['representation'].iloc[0]
        subset = dataset_subset[dataset_subset['representation'] == first_rep]
        
        # Sum rows and missing_cells across splits
        total_rows = subset['rows'].sum()
        total_missing_cells = subset['missing_cells'].sum()
        
        # Features and classes should be constant per dataset, so take the first value
        # Add 1 to features to account for the classification feature
        features = subset['features'].iloc[0] + 1
        classes = subset['classes'].iloc[0]
        
        # Calculate tensor size Ω using feature cardinalities
        tensor_size = None
        if cardinalities_df is not None:
            # Filter for this dataset and tensor representation only
            dataset_cards = cardinalities_df[
                (cardinalities_df['dataset'] == name) & 
                (cardinalities_df['representation'] == 'tensor')
            ]
            if not dataset_cards.empty:
                # Convert cardinality to numeric and remove NaN values
                cardinalities = pd.to_numeric(
                    dataset_cards['cardinality'], 
                    errors='coerce'
                ).dropna().values
                if len(cardinalities) > 0:
                    # Tensor size Ω = product of all cardinalities * number of classes
                    tensor_size = int(np.prod(cardinalities) * classes)
        
        # Fallback to old calculation if no cardinalities available
        if tensor_size is None:
            tensor_size = total_rows * features
        
        # Calculate sparsity and non-zero entries
        sparsity = total_rows / tensor_size if tensor_size > 0 else 0
        
        results_list.append({
            'dataset': name,
            'features': features,
            'non_zero_entries': total_rows,
            'tensor_size': tensor_size,
            'sparsity': sparsity,
            'classes': classes
        })
    
    return pd.DataFrame(results_list)
    

# add argument for the path to the results file
if __name__ == "__main__":    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        results = pd.read_csv(results_path)
        
        # Determine cardinalities path (default to feature_cardinalities.csv in data/reports/)
        cardinalities_path = sys.argv[2] if len(sys.argv) > 2 else "src/data_pipeline/data/reports/feature_cardinalities.csv"
        
        metrics = calculate_metrics(results, cardinalities_path)
        
        # Save to CSV in src/visualization folder
        output_path = Path(__file__).parent / "dataset_metrics.csv"
        metrics.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")
        print(metrics)
    else:
        print("Usage: python visualize_results.py <path_to_results_csv> [path_to_cardinalities_csv]")


# Example usage:
# python visualize_results.py src/data_pipeline/data/reports/dataset_results.csv src/data_pipeline/data/reports/feature_cardinalities.csv
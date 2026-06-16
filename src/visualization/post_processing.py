import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from src.data_pipeline.make_dataset import DATASET_CONFIGS
import argparse

def generate_plots(results: pd.DataFrame, output_dir: str):
    """
    Generate box plots for the number of rows per dataset (for now)

    Args:
        results (pd.DataFrame): DataFrame containing the results with a 'dataset' column and a 'non_zero_entries' column.
        output_dir (str): Directory where the generated box plots will be saved.
    """

    # ensure that the output directory exists
    os.makedirs(output_dir, exist_ok = True)

    # (for now) use the dataset_metrics.csv file to get the number of rows per dataset
    if results['dataset'].isin(DATASET_CONFIGS.keys()).any():
        metrics = results[['dataset', 'non_zero_entries']].copy()
    else:
        raise ValueError("Dataset names in results do not match any configured datasets.")

    # Generate box plot for non-zero entries per dataset
    plt.figure(figsize=(10, 6))
    metrics.boxplot(column='non_zero_entries', by='dataset')
    plt.title('Box Plot of Non-Zero Entries per Dataset')
    plt.suptitle('')  # Suppress the default title to avoid redundancy
    plt.xlabel('Dataset')
    plt.ylabel('# Non-Zero Entries')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots/non_zero_entries_boxplot.png'))
    plt.close()

def calculate_table_metrics(results: pd.DataFrame, cardinalities: str | None, output_dir: str) -> pd.DataFrame:
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
    # Ensure that the output directory exists
    os.makedirs(output_dir, exist_ok = True)

    # Load cardinalities if path is provided
    cardinalities_df = None

    if cardinalities is not None:
        cardinalities_df = pd.read_csv(cardinalities)
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
        sparsity = total_missing_cells / tensor_size if tensor_size > 0 else 0
        
        results_list.append({
            'dataset': name,
            'features': features,
            'non_zero_entries': total_rows,
            'tensor_size': tensor_size,
            'sparsity': sparsity,
            'classes': classes
        })
    
    return pd.DataFrame(results_list)

def plot_all_epoch_metrics(metrics_path: str, output_dir: str): 
    """
    Finds all metrics.csv files recursively, compresses multi-row epoch logs, 
    and handles plotting.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(metrics_path):
        for filename in files:
            if filename == "metrics.csv":
                file_path = os.path.join(root, filename)
                
                dataset_and_seed = os.path.basename(root)
                model_name = os.path.basename(os.path.dirname(root))
                
                if "_seed" not in dataset_and_seed:
                    continue

                print(f"Processing: Model={model_name}, Dataset/Seed={dataset_and_seed}")

                try:
                    # 1. Load data
                    raw_df = pd.read_csv(file_path)
                    
                    # 2. Collapse alternating rows by grouping by epoch and step
                    # .first() ignores the NaN values and grabs the actual numeric log entry
                    results_df = raw_df.groupby(['epoch', 'step'], as_index=False).first()
                    
                    # 3. Setup Plot
                    plt.figure(figsize=(10, 6))
                    
                    # Target metrics present in your actual CSV snippet
                    metrics_to_plot = results_df.columns
                    
                    for metric in metrics_to_plot:
                        if metric in results_df.columns and not results_df[metric].isna().all():
                            # Use 'epoch' as the X-axis
                            plt.plot(results_df["epoch"], results_df[metric], marker='o', label=metric)
                        else:
                            print(f"Warning: {metric} contains no data or wasn't found in {file_path}")
                    
                    plt.xlabel("Epoch")
                    plt.ylabel("Value")
                    plt.title(f"Metrics for {model_name} - {dataset_and_seed}")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.legend()
                    
                    # 4. Save and clean up
                    # Construct the dynamic path to the dataset-specific folder
                    dataset_plot_dir = os.path.join(output_dir, "plots", dataset_and_seed)
                    
                    # Automatically create the folder on the fly if it doesn't exist yet
                    os.makedirs(dataset_plot_dir, exist_ok=True)
                    
                    # Define the final file path inside that newly created folder
                    save_path = os.path.join(dataset_plot_dir, f"{model_name}_{dataset_and_seed}.png")
                    
                    # Save and clear the plot memory
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()     
                    
                except Exception as e:
                    import traceback
                    print(f"Error reading {file_path}: {e}")
                    traceback.print_exc()
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate dataset metrics and generate visualizations."
    )

    parser.add_argument(
        "--function",
        dest="function_name",
        required=True,
        choices=["calculate_table_metrics", "generate_plots", "plot_all_epoch_metrics"],
        help="Name of the function to execute."
    )

    parser.add_argument(
        "--results",
        dest="results_file",
        default="src/visualization/dataset_metrics.csv",
        help="Path to the CSV file containing the results."
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="src/visualization",
        help="Directory where the generated plots will be saved."
    )

    parser.add_argument(
        "--metrics-path",
        dest="metrics_path",
        default="src/summary_results/results",
        help="Path to recursively search for metrics.csv files used by plot_all_epoch_metrics."
    )

    parser.add_argument(
        "--cardinalities",
        dest="cardinalities",
        default=None,
        help="Path to a cardinalities CSV file used by calculate_table_metrics."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    results = pd.read_csv(args.results_file)

    if args.function_name == "calculate_table_metrics":
        metrics = calculate_table_metrics(
            results=results,
            cardinalities=args.cardinalities,
            output_dir=args.output_dir,
        )

        metrics.to_csv(
            os.path.join(args.output_dir, "dataset_metrics.csv"),
            index=False,
        )

    elif args.function_name == "generate_plots":
        generate_plots(
            results=results,
            output_dir=args.output_dir,
        )

    elif args.function_name == "plot_all_epoch_metrics":
        plot_all_epoch_metrics(
            metrics_path=args.metrics_path,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()

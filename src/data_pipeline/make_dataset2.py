# %% Imports
import argparse
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from src.data_pipeline.dataset_configs import DATASET_CONFIGS
from src.data_pipeline.preprocessing2 import (
    clean_targets,
    fit_preprocessor,
    load_raw_dataset_splits,
    pool_dataset_splits,
    remove_rare_classes,
    save_metadata,
    save_splits,
    split_tuning_and_folds,
    transform_dataset,
)

# %% CLI options

def parse_args():
    parser = argparse.ArgumentParser(description="Create processed train, validation, and test splits.")

    parser.add_argument(
        "--dataset",
        dest="dataset_name",
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Name of the dataset configuration to use.",
    )
    parser.add_argument(
        "--representation",
        choices=["baseline", "tensor", "both"],
        default="both",
        help="Which processed representation to create.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("src/data_pipeline/data/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data_pipeline/data/processed"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Number of bins used when discretizing numerical features for tensor representation.",
    )
    parser.add_argument(
        "--binning-strategy",
        choices=["quantile", "uniform"],
        default="quantile",
    )

    return parser.parse_args()

# %% Setup

def main():
    args = parse_args() # reads whatever flags were passed on the command line
    logger = logging.getLogger(__name__)
    config = DATASET_CONFIGS[args.dataset_name] # looks up that dataset's settings

    logger.info("Processing dataset: %s", args.dataset_name)

    # load whatever raw files exist, then combine them into one pooled dataframe
    raw_train_source, raw_test_source = load_raw_dataset_splits(config=config, raw_dir=args.raw_dir)
    pooled_df = pool_dataset_splits(raw_train_source, raw_test_source)

    target_column = config["target_column"]

    # cleans up
    pooled_df = clean_targets(pooled_df, target_column=target_column)
    pooled_df = remove_rare_classes(pooled_df, target_column=target_column)

    tuning_train_df, tuning_val_df, folds = split_tuning_and_folds(pooled_df, target_column = target_column,random_state = args.seed)
                                                                     
    if args.representation == "both":
        representations = ["baseline", "tensor"]
    else:
        representations = [args.representation]

    # fit and transform
    for rep in representations:
        metadata = fit_preprocessor( # called once per representation
            df=pooled_df,
            config=config,
            representation=rep,
            n_bins=args.n_bins,
            binning_strategy=args.binning_strategy,
        )
    
        transformed_tuning_train = transform_dataset(tuning_train_df, metadata)
        transformed_tuning_val = transform_dataset(tuning_val_df, metadata)

        transformed_folds = []
        for fold_train_df, fold_test_df in folds:
            transformed_fold_train = transform_dataset(fold_train_df, metadata)
            transformed_fold_test = transform_dataset(fold_test_df, metadata)
            transformed_folds.append((transformed_fold_train, transformed_fold_test))
        
        split_paths = save_splits(
            tuning_train_df=transformed_tuning_train,
            tuning_val_df=transformed_tuning_val,
            folds=transformed_folds,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            representation=rep,
        )

        metadata.update({
            "dataset_name": args.dataset_name,
            "task": config.get("task", "classification"),
            "seed": args.seed,
            "n_tuning_train": len(transformed_tuning_train),
            "n_tuning_val": len(transformed_tuning_val),
        })

        for fold_number, (fold_train_df, fold_test_df) in enumerate(transformed_folds, start=1):
            metadata[f"n_fold_{fold_number}_train"] = len(fold_train_df)
            metadata[f"n_fold_{fold_number}_test"] = len(fold_test_df)

        metadata_path = save_metadata(
            metadata=metadata,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            representation=rep,
        )

        logger.info("Saved %s tuning_train split: %s", rep, split_paths["tuning_train"])
        logger.info("Saved %s tuning_val split: %s", rep, split_paths["tuning_val"])
        logger.info("Saved %s metadata: %s", rep, metadata_path)
    logger.info("Done.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
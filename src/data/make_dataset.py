# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.data.dataset_configs import DATASET_CONFIGS
from src.data.preprocessing import (
    load_raw_dataset,
    process_dataset,
    save_metadata,
    save_splits,
    split_dataset,
)


@click.command()
@click.option(
    "--dataset",
    "dataset_name",
    required=True,
    type=click.Choice(sorted(DATASET_CONFIGS.keys())),
    help="Name of the dataset configuration to use.",
)
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    show_default=True,
    help="Directory containing raw dataset files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
    help="Directory where processed files will be saved.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed used for train/val/test splits.",
)
def main(
    dataset_name: str,
    raw_dir: Path,
    output_dir: Path,
    seed: int,
) -> None:
    """
    Process a raw tabular dataset into train/val/test CSV files
    and save metadata needed for downstream modeling.
    """
    logger = logging.getLogger(__name__)
    config = DATASET_CONFIGS[dataset_name]

    logger.info("Processing dataset: %s", dataset_name)
    logger.info("Loading raw data from: %s", raw_dir / config["file_name"])

    raw_df = load_raw_dataset(config=config, raw_dir=raw_dir)
    logger.info("Raw shape: %s", raw_df.shape)

    processed_df, metadata = process_dataset(df=raw_df, config=config)
    logger.info("Processed shape after cleaning/encoding: %s", processed_df.shape)

    train_df, val_df, test_df = split_dataset(
        df=processed_df,
        target_column="target",
        random_state=seed,
    )

    split_paths = save_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    metadata.update(
        {
            "dataset_name": dataset_name,
            "task": config.get("task", "classification"),
            "seed": seed,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        }
    )

    metadata_path = save_metadata(
        metadata=metadata,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    logger.info("Saved train split: %s", split_paths["train"])
    logger.info("Saved val split: %s", split_paths["val"])
    logger.info("Saved test split: %s", split_paths["test"])
    logger.info("Saved metadata: %s", metadata_path)
    logger.info("Done.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()
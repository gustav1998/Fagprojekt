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
    "--representation",
    type=click.Choice(["baseline", "tensor", "both"]),
    default="both",
    show_default=True,
    help="Which processed representation to create.",
)
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
)
@click.option(
    "--n-bins",
    type=int,
    default=None,
    help="Number of bins used when discretizing numerical features for tensor representation.",
)
@click.option(
    "--binning-strategy",
    type=click.Choice(["quantile", "uniform"]),
    default="quantile",
    show_default=True,
)
def main(
    dataset_name: str,
    representation: str,
    raw_dir: Path,
    output_dir: Path,
    seed: int,
    n_bins: int | None,
    binning_strategy: str,
) -> None:
    """Kør pipeline for at opbygge og gemme forbehandlede datasæt.

    Skriptet:
    - læser råt data
    - processerer (encode, discretize)
    - splitter i train/val/test
    - gemmer CSV-splits og metadata
    """
    logger = logging.getLogger(__name__)
    config = DATASET_CONFIGS[dataset_name]

    logger.info("Processing dataset: %s", dataset_name)
    raw_df = load_raw_dataset(config=config, raw_dir=raw_dir)
    logger.info("Raw shape: %s", raw_df.shape)

    representations = ["baseline", "tensor"] if representation == "both" else [representation]

    for rep in representations:
        logger.info("Creating representation: %s", rep)

        processed_df, metadata = process_dataset(
            df=raw_df,
            config=config,
            representation=rep,
            n_bins=n_bins,
            binning_strategy=binning_strategy,
        )

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
            representation=rep,
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
            representation=rep,
        )

        logger.info("Saved %s train split: %s", rep, split_paths["train"])
        logger.info("Saved %s val split: %s", rep, split_paths["val"])
        logger.info("Saved %s test split: %s", rep, split_paths["test"])
        logger.info("Saved %s metadata: %s", rep, metadata_path)

    logger.info("Done.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
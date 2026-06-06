"""Command-line training script for tabular classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from src.data.datamodule import TabularDataModule
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLPClassifier
from src.models.lightning_module import TabularClassifierModule
from src.models.cpd import CPDClassifier
from src.models.tt import TTClassifier
from src.models.tr import TRClassifier


DEFAULT_TRAINING_CONFIGS = {
    "lr": {
        "epochs": 100,
        "learning_rate": 1e-3,
    },
    "mlp": {
        "epochs": 100,
        "learning_rate": 1e-3,
    },
    "cpd": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "rank": 16,
    },
    "tt": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "rank": 16,
    },
    "tr": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "rank": 16,
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lr", "mlp", "cpd", "tt", "tr"],
        help=(
            "Model to train (lr=logistic, mlp=neural net, "
            "cpd/tt/tr=tensor classifiers)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed dataset splits.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Defaults depend on model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate. Defaults depend on model.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Lightning accelerator",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Tensor rank for CPD, TT, and TR. Defaults depend on model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model initialization and data shuffling.",
    )
    parser.add_argument(
        "--result-version",
        type=str,
        default=None,
        help="Result folder name under results/<model>/.",
    )

    return parser.parse_args()


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset hyperparameters from model-specific defaults."""
    defaults = DEFAULT_TRAINING_CONFIGS[args.model]

    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.learning_rate is None:
        args.learning_rate = defaults["learning_rate"]
    if args.rank is None:
        args.rank = defaults.get("rank", 8)

    return args


def main() -> None:
    """Run training and testing for one model on one dataset."""
    args = apply_model_defaults(parse_args())
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    datamodule = TabularDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        model_type=args.model,
        processed_dir=args.processed_dir,
        seed=args.seed,
    )
    datamodule.setup()

    if args.model == "lr":
        base_model = LogisticRegression(
            input_dim=datamodule.input_dim,
            num_classes=datamodule.num_classes,
        )

    elif args.model == "mlp":
        base_model = MLPClassifier(
            input_dim=datamodule.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=datamodule.num_classes,
            dropout=args.dropout,
        )

    elif args.model == "cpd":
        base_model = CPDClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )

    elif args.model == "tt":
        base_model = TTClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )

    elif args.model == "tr":
        base_model = TRClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = TabularClassifierModule(
        model=base_model,
        learning_rate=args.learning_rate,
    )

    result_version = args.result_version
    if result_version is None:
        result_version = (
            f"{args.dataset}_seed{args.seed}"
            if args.seed is not None
            else args.dataset
        )

    logger = CSVLogger(
        save_dir="results",
        name=args.model,
        version=result_version,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

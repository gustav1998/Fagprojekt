"""Command-line training script for tabular classifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.datamodule import TabularDataModule
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLPClassifier
from src.models.lightning_module import TabularClassifierModule
from src.models.cpd import CPDClassifier
from src.models.mba import MBAClassifier
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
    "mba": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "interaction_order": 3,
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


def infer_monitor_mode(metric: str) -> str:
    """Infer whether a validation metric should be minimized or maximized."""
    return "min" if metric.endswith("loss") else "max"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lr", "mlp", "cpd", "mba", "tt", "tr"],
        help=(
            "Model to train (lr=logistic, mlp=neural net, "
            "cpd/mba/tt/tr=tensor classifiers)"
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
        "--interaction-order",
        type=int,
        default=None,
        help="Maximum MBA interaction order. Defaults depend on model.",
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
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        default=True,
        help="Use early stopping and test the best validation checkpoint.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Train for all epochs and test the final model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience in validation epochs.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="Validation metric monitored by checkpointing/early stopping.",
    )
    parser.add_argument(
        "--monitor-mode",
        choices=["min", "max"],
        default=None,
        help="Whether the monitored metric should be minimized or maximized.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only train/validate. Useful for hyperparameter tuning.",
    )

    return parser.parse_args()


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace: # checks whether any model-specific hyperparameters are unset and fills them in from defaults
    """Fill unset hyperparameters from model-specific defaults."""
    defaults = DEFAULT_TRAINING_CONFIGS[args.model]

    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.learning_rate is None:
        args.learning_rate = defaults["learning_rate"]
    if args.rank is None:
        args.rank = defaults.get("rank", 8)
    if args.interaction_order is None:
        args.interaction_order = defaults.get("interaction_order", 3)

    return args


def main() -> None:
    """Run training and testing for one model on one dataset."""
    args = apply_model_defaults(parse_args())
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True) # check whether a seed was provided and if so, set it for reproducibility

    datamodule = TabularDataModule(
        dataset_name = args.dataset,
        batch_size = args.batch_size,
        model_type = args.model,
        processed_dir = args.processed_dir,
        seed = args.seed,
    )
    datamodule.setup()

    # check which model was selected and initialize it with the appropriate hyperparameters and data dimensions:
    if args.model == "lr":
        base_model = LogisticRegression(
            input_dim = datamodule.input_dim,
            num_classes = datamodule.num_classes,
        )

    elif args.model == "mlp":
        base_model = MLPClassifier(
            input_dim = datamodule.input_dim,
            hidden_dim = args.hidden_dim,
            num_classes = datamodule.num_classes,
            dropout = args.dropout,
        )

    elif args.model == "cpd":
        base_model = CPDClassifier(
            feature_dims = datamodule.cardinalities,
            rank = args.rank,
            num_classes = datamodule.num_classes,
        )

    elif args.model == "mba":
        base_model = MBAClassifier(
            feature_dims = datamodule.cardinalities,
            interaction_order = args.interaction_order,
            num_classes = datamodule.num_classes,
        )

    elif args.model == "tt":
        base_model = TTClassifier(
            feature_dims = datamodule.cardinalities,
            rank = args.rank,
            num_classes = datamodule.num_classes,
        )

    elif args.model == "tr":
        base_model = TRClassifier(
            feature_dims = datamodule.cardinalities,
            rank = args.rank,
            num_classes = datamodule.num_classes,
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}") 

    model = TabularClassifierModule(
        model = base_model,
        learning_rate = args.learning_rate,
        num_classes = datamodule.num_classes,
    )

    result_version = args.result_version
    if result_version is None:
        result_version = (
            f"{args.dataset}_seed{args.seed}"
            if args.seed is not None
            else args.dataset
        )

    # saves results:
    logger = CSVLogger(
        save_dir = "results",
        name = args.model,
        version = result_version,
    )

    monitor_mode = args.monitor_mode or infer_monitor_mode(args.monitor)
    callbacks = []
    checkpoint_callback = None
    if args.early_stopping:
        checkpoint_callback = ModelCheckpoint(
            dirpath = Path(logger.log_dir) / "checkpoints",
            filename = "best",
            monitor = args.monitor,
            mode = monitor_mode,
            save_top_k = 1,
        )
        callbacks.extend(
            [
                checkpoint_callback,
                EarlyStopping(
                    monitor = args.monitor,
                    mode = monitor_mode,
                    patience = args.patience,
                ),
            ]
        )

    # initialize the Lightning trainer with the specified accelerator, logger, and callbacks and run training and testing:
    trainer = L.Trainer(
        max_epochs = args.epochs,
        accelerator = args.accelerator,
        devices = 1,
        logger = logger,
        enable_checkpointing = args.early_stopping,
        callbacks = callbacks,
        log_every_n_steps = 1,
    )

    start_time = time.perf_counter() 

    trainer.fit(model, datamodule=datamodule)

    fit_seconds = time.perf_counter() - start_time # time the training process

    test_seconds = None
    if not args.skip_test:

        test_start_time = time.perf_counter()

        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path="best" if args.early_stopping else None,
        )
        test_seconds = time.perf_counter() - test_start_time # time the testing process

    # logs and saves metadata about the training run to a JSON file in the result folder
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "rank": args.rank if args.model in {"cpd", "tt", "tr"} else None,
        "interaction_order": (
            args.interaction_order if args.model == "mba" else None
        ),
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
        "early_stopping": args.early_stopping,
        "patience": args.patience if args.early_stopping else None,
        "monitor": args.monitor if args.early_stopping else None,
        "monitor_mode": monitor_mode if args.early_stopping else None,
        "best_model_path": (
            checkpoint_callback.best_model_path
            if checkpoint_callback is not None
            else None
        ),
        "best_model_score": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback is not None
            and checkpoint_callback.best_model_score is not None
            else None
        ),
        "stopped_epoch": trainer.current_epoch,
        "skip_test": args.skip_test,
        "num_classes": datamodule.num_classes,
        "input_dim": datamodule.input_dim,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
    }
    metadata_path = Path(logger.log_dir) / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

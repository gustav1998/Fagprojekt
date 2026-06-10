"""Command-line training script for tabular classifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.ensemble import RandomForestClassifier

from src.data.datamodule import TabularDataModule
from src.data.load_processed import load_processed_dataset
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLPClassifier
from src.models.lightning_module import TabularClassifierModule
from src.models.cpd import CPDClassifier
from src.models.mba import MBAClassifier
from src.models.tt import TTClassifier
from src.models.tr import TRClassifier
from src.models.rf import DEFAULT_RF_CONFIG, compute_metrics, parse_max_features


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
        choices=["lr", "mlp", "cpd", "mba", "tt", "tr", "rf"],
        help=(
            "Model to train (lr=logistic, mlp=neural net, "
            "cpd/mba/tt/tr=tensor classifiers, rf=random forest)"
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
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_RF_CONFIG["max_depth"],
        help="RF only: maximum tree depth. Omit for unlimited.",
    )
    parser.add_argument(
        "--max-features",
        type=parse_max_features,
        default=DEFAULT_RF_CONFIG["max_features"],
        help='RF only: features per split ("sqrt", "log2", or float fraction).',
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=DEFAULT_RF_CONFIG["min_samples_leaf"],
        help="RF only: minimum samples required at a leaf node.",
    )

    return parser.parse_args()


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace: # checks whether any model-specific hyperparameters are unset and fills them in from defaults
    """Fill unset hyperparameters from model-specific defaults."""
    if args.model == "rf":
        return args
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


def _train_rf(args: argparse.Namespace, result_version: str) -> None:
    """Train a Random Forest and save results in the standard format."""
    data = load_processed_dataset(
        dataset_name=args.dataset,
        representation="baseline",
        processed_dir=args.processed_dir,
    )

    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]
    metadata = data["metadata"]

    X_train = train_df.drop(columns=["target"]).to_numpy()
    y_train = train_df["target"].to_numpy()
    X_val = val_df.drop(columns=["target"]).to_numpy()
    y_val = val_df["target"].to_numpy()
    X_test = test_df.drop(columns=["target"]).to_numpy()
    y_test = test_df["target"].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=DEFAULT_RF_CONFIG["n_estimators"],
        max_depth=args.max_depth,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )

    fit_start = time.perf_counter()
    rf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    val_metrics = compute_metrics(y_val, rf.predict(X_val))

    test_metrics: dict = {}
    test_seconds = None
    if not args.skip_test:
        test_start = time.perf_counter()
        test_metrics = compute_metrics(y_test, rf.predict(X_test))
        test_seconds = time.perf_counter() - test_start

    result_dir = Path("results") / "rf" / result_version
    result_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "epoch": 0,
        "step": 0,
        "val_acc": val_metrics["acc"],
        "val_balanced_acc": val_metrics["balanced_acc"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "test_acc": test_metrics.get("acc"),
        "test_loss": None,
        "test_balanced_acc": test_metrics.get("balanced_acc"),
        "test_macro_precision": test_metrics.get("macro_precision"),
        "test_macro_recall": test_metrics.get("macro_recall"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_weighted_f1": test_metrics.get("weighted_f1"),
    }
    pd.DataFrame([row]).to_csv(result_dir / "metrics.csv", index=False)

    if not args.skip_test:
        joblib.dump(rf, result_dir / "model.joblib")

    run_metadata = {
        "dataset": args.dataset,
        "model": "rf",
        "seed": args.seed,
        "n_estimators": DEFAULT_RF_CONFIG["n_estimators"],
        "max_depth": args.max_depth,
        "max_features": args.max_features,
        "min_samples_leaf": args.min_samples_leaf,
        "num_classes": len(metadata["target_mapping"]),
        "input_dim": X_train.shape[1],
        "trainable_parameters": None,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
    }
    (result_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved results to {result_dir}")
    print(json.dumps(run_metadata, indent=2))


def main() -> None:
    """Run training and testing for one model on one dataset."""
    args = apply_model_defaults(parse_args())
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True) # check whether a seed was provided and if so, set it for reproducibility

    result_version = args.result_version or (
        f"{args.dataset}_seed{args.seed}"
        if args.seed is not None
        else args.dataset
    )

    if args.model == "rf":
        _train_rf(args, result_version)
        return

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

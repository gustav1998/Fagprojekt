# %% 
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_pipeline.datamodule2 import TabularDataModule
from src.data_pipeline.load_processed2 import load_processed_dataset
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLPClassifier
from src.models.lightning_module import TabularClassifierModule
from src.models.cpd2 import ClassParafacClassifier
from src.models.mba2 import MBAClassifier
from src.models.tt2 import ClassTTClassifier
from src.models.tt3 import ClassTTUniformClassifier
from src.models.tr2 import ClassTRClassifier
from src.models.rf import DEFAULT_RF_CONFIG, compute_metrics, parse_max_features

# %% default hyperparameters used when none are provided via command line. Tuned values from tune_hyperparameters2 override these

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
        "interaction_order": 2,
    },
    "tt": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "rank": 16,
    },
    "tt3": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "rank": 2,
    },
    "tr": {
        "epochs": 60,
        "learning_rate": 1e-2,
        "rank": 16,
    },
}

# %%

def infer_monitor_mode(metric: str): # infers whether a validation metric should be minimized or maximized 
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
        choices=["lr", "mlp", "cpd", "mba", "tt", "tt3", "tr", "rf"],
        help=(
            "Model to train (lr=logistic, mlp=neural net, "
            "cpd/mba/tt/tt3/tr=tensor classifiers, rf=random forest)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of DataLoader worker processes. The default is 0 because "
            "the processed data is already loaded into memory."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("src/data_pipeline/data/processed"),
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
        help="Tensor rank for CPD, TT, TT3, and TR. Defaults depend on model.",
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
        help="Result folder name under src/summary_results/results/<model>/.",
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
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tuning", "fold"],
        help="Whether to train on the tuning split or a CV fold.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number (1-5). Required when mode is 'fold'.",
    )

    return parser.parse_args()

# %% Checks whether any model-specific hyperparameters are unset and fills them in from defaults
def apply_model_defaults(args):
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

# %% Builds the name of the results folder

def build_result_version(args):
    if args.result_version is not None:
        return args.result_version

    if args.mode == "tuning":
        base = f"{args.dataset}_tuning"
    else:
        base = f"{args.dataset}_fold{args.fold}"

    if args.seed is not None:
        base = base + f"_seed{args.seed}"

    return base

# %% trains the random forest and saves results

def _train_rf(args, result_version):
    data = load_processed_dataset(
        dataset_name=args.dataset,
        representation="baseline",
        processed_dir=args.processed_dir,
        mode=args.mode,
        fold=args.fold,
    )

    if args.mode == "tuning": # in tuning mode, tuning_val is used for both validation and test
        train_df = data["train_df"]
        val_df = data["val_df"]
        test_df = data["val_df"]
    else:
        fold_test_df = data["val_df"] # val_df from load_processed2 is the held-out fold test set
        train_indices, val_indices = train_test_split( # split fold_train 80/20 internally so RF has a validation set for early stopping
            range(len(data["train_df"])),
            test_size=0.2,
            random_state=args.seed,
        )
        train_df = data["train_df"].iloc[train_indices].reset_index(drop=True)
        val_df = data["train_df"].iloc[val_indices].reset_index(drop=True)
        test_df = fold_test_df # the actual held-out fold is used as the test set

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
    fit_seconds = time.perf_counter() - fit_start # time the training process

    val_metrics = compute_metrics(y_val, rf.predict(X_val)) # evaluate on internal val set (not the held-out fold test)

    test_metrics = {}
    test_seconds = None
    if not args.skip_test: # evaluate on the held-out fold test set and time it
        test_start = time.perf_counter()
        test_metrics = compute_metrics(y_test, rf.predict(X_test))
        test_seconds = time.perf_counter() - test_start

    result_dir = Path("src/summary_results/results") / "rf" / result_version
    result_dir.mkdir(parents=True, exist_ok=True)

    row = { # saves val and test metrics in the same format as the lightning models so summarize_results can read both
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
        "mode": args.mode,
        "fold": args.fold,
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

# %%

def main():
    args = apply_model_defaults(parse_args())
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers.*",
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*treespec, LeafSpec.*deprecated.*treespec, TreeSpec.*is_leaf.*",
        category=Warning,
    )

    if args.mode == "fold" and args.fold is None:
        raise ValueError("--fold must be specified when --mode is 'fold'")

    processed_dir = args.processed_dir
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True) # check whether a seed was provided and if so, set it for reproducibility

    result_version = build_result_version(args)

    if args.model == "rf": # rf has its own training path that doesn't use lightning
        _train_rf(args, result_version)
        return

    datamodule = TabularDataModule( # loads the correct splits and prepares dataloaders based on mode and fold
        dataset_name=args.dataset,
        mode=args.mode,
        fold=args.fold,
        batch_size=args.batch_size,
        model_type=args.model,
        processed_dir=processed_dir,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    datamodule.setup()

    # check which model was selected and initialize it with the appropriate hyperparameters and data dimensions:
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
        base_model = ClassParafacClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )
    elif args.model == "mba":
        base_model = MBAClassifier(
            feature_dims=datamodule.cardinalities,
            interaction_order=args.interaction_order,
            num_classes=datamodule.num_classes,
        )
    elif args.model == "tt":
        base_model = ClassTTClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )
    elif args.model == "tt3":
        base_model = ClassTTUniformClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )
    elif args.model == "tr":
        base_model = ClassTRClassifier(
            feature_dims=datamodule.cardinalities,
            rank=args.rank,
            num_classes=datamodule.num_classes,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = TabularClassifierModule(
        model=base_model,
        learning_rate=args.learning_rate,
        num_classes=datamodule.num_classes,
    )

    # delete stale metrics.csv before creating the logger to avoid CSVLogger append bug
    stale_metrics = Path("src/summary_results/results") / args.model / result_version / "metrics.csv"
    if stale_metrics.exists():
        stale_metrics.unlink()

    # saves results:
    logger = CSVLogger(
        save_dir="src/summary_results/results",
        name=args.model,
        version=result_version,
    )

    monitor_mode = args.monitor_mode or infer_monitor_mode(args.monitor) # infer min/max from metric name if not explicitly set
    callbacks = []
    checkpoint_callback = None
    if args.early_stopping: # saves the best checkpoint and stops early if the monitored metric stops improving
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "checkpoints",
            filename="best",
            monitor=args.monitor,
            mode=monitor_mode,
            save_top_k=1,
        )
        callbacks.extend([
            checkpoint_callback,
            EarlyStopping(
                monitor=args.monitor,
                mode=monitor_mode,
                patience=args.patience,
            ),
        ])

    # initialize the Lightning trainer with the specified accelerator, logger, and callbacks and run training and testing:
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        enable_checkpointing=args.early_stopping,
        callbacks=callbacks,
        log_every_n_steps=1,
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
    run_metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "rank": args.rank if args.model in {"cpd", "tt", "tt3", "tr"} else None,
        "interaction_order": args.interaction_order if args.model == "mba" else None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "early_stopping": args.early_stopping,
        "patience": args.patience if args.early_stopping else None,
        "monitor": args.monitor if args.early_stopping else None,
        "monitor_mode": monitor_mode if args.early_stopping else None,
        "best_model_path": checkpoint_callback.best_model_path if checkpoint_callback is not None else None,
        "best_model_score": float(checkpoint_callback.best_model_score) if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None else None,
        "stopped_epoch": trainer.current_epoch,
        "skip_test": args.skip_test,
        "num_classes": datamodule.num_classes,
        "input_dim": datamodule.input_dim,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
    }
    metadata_path = Path(logger.log_dir) / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

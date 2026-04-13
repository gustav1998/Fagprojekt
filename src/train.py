from __future__ import annotations

import argparse

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from src.data.datamodule import TabularDataModule
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLPClassifier
from src.models.lightning_module import TabularClassifierModule
from src.models.cpd import CPDClassifier

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lr", "mlp", "cpd"],
        help="Model to train",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Lightning accelerator",
    )
    parser.add_argument("--rank", type=int, default=8, help="CP rank for CPD model")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    datamodule = TabularDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        model_type=args.model,
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

    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    model = TabularClassifierModule(
        model=base_model,
        learning_rate=args.learning_rate,
    )

    logger = CSVLogger(
        save_dir="results",
        name=args.model,
        version=f"{args.dataset}",
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
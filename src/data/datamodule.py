from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_processed import load_processed_dataset, dataframe_to_tensors
from src.utils.encoding import one_hot_encode_features


class TabularDataModule(L.LightningDataModule):
    """LightningDataModule til tabulære datasæt.

    Denne klasse indlæser de forbehandlede CSV-filer, konverterer til tensorer
    og returnerer PyTorch `DataLoader`-objekter til træning, validering og test.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 256,
        model_type: str = "lr",
        processed_dir: str = "data/processed",
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.model_type = model_type
        self.processed_dir = processed_dir
        self.num_workers = num_workers

        # Metadata om dataset (kolonnenavne, cardinalities osv.)
        self.metadata = None
        self.cardinalities = None

        # Tensors for splits
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        # Model relaterede størrelser (beregnes i setup)
        self.input_dim = None
        self.num_classes = None

    def setup(self, stage: str | None = None) -> None:
        """Hent forbehandlede data og forbered tensorer.

        `representation` vælger om vi bruger one-hot (baseline) eller tensor-encoded
        (f.eks. embeddings) repræsentation, afhængigt af modeltypen.
        """
        representation = "baseline" if self.model_type in {"lr", "mlp"} else "tensor"

        data = load_processed_dataset(
            dataset_name=self.dataset_name,
            representation=representation,
            processed_dir=self.processed_dir,
        )

        self.metadata = data["metadata"]
        self.cardinalities = self.metadata["cardinalities"]

        # Konverter pandas -> torch tensor
        X_train_raw, y_train = dataframe_to_tensors(data["train_df"], device="cpu")
        X_val_raw, y_val = dataframe_to_tensors(data["val_df"], device="cpu")
        X_test_raw, y_test = dataframe_to_tensors(data["test_df"], device="cpu")

        # For baseline repræsentation: one-hot encode kategoriske features
        if self.model_type in {"lr", "mlp"}:
            categorical_cardinalities = self.metadata["categorical_cardinalities"]
            num_numerical_features = len(self.metadata["numerical_columns"])

            self.X_train = one_hot_encode_features(
                X_train_raw,
                categorical_cardinalities=categorical_cardinalities,
                num_numerical_features=num_numerical_features,
            )
            self.X_val = one_hot_encode_features(
                X_val_raw,
                categorical_cardinalities=categorical_cardinalities,
                num_numerical_features=num_numerical_features,
            )
            self.X_test = one_hot_encode_features(
                X_test_raw,
                categorical_cardinalities=categorical_cardinalities,
                num_numerical_features=num_numerical_features,
            )
        elif self.model_type in {"cpd", "mba"}:
            # For tensor-baserede modeller holdes kategorier som indices
            self.X_train = X_train_raw.long()
            self.X_val = X_val_raw.long()
            self.X_test = X_test_raw.long()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # Input-dimension til modeller og antal klasser
        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(self.metadata["target_mapping"])

    def train_dataloader(self) -> DataLoader:
        """Returner trænings-DataLoader."""
        dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returner validerings-DataLoader."""
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returner test-DataLoader."""
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
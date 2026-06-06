from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_processed import (
    dataframe_to_tensors,
    load_processed_dataset,
)
from src.utils.encoding import one_hot_encode_features


ONE_HOT_MODELS = {"lr", "mlp"}
TENSOR_MODELS = {"cpd", "mba", "tt", "tr"}


def representation_for_model(model_type: str) -> str:
    """Return the processed data representation expected by a model."""
    if model_type in ONE_HOT_MODELS:
        return "baseline"
    if model_type in TENSOR_MODELS:
        return "tensor"
    raise ValueError(f"Unsupported model type: {model_type}")


class TabularDataModule(L.LightningDataModule):
    """Load processed tabular data into PyTorch DataLoaders."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 256,
        model_type: str = "lr",
        processed_dir: str = "data/processed",
        num_workers: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.model_type = model_type
        self.processed_dir = processed_dir
        self.num_workers = num_workers
        self.seed = seed

        self.metadata = None
        self.cardinalities = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.input_dim = None
        self.num_classes = None

    def setup(self, stage: str | None = None) -> None:
        """Load processed splits and convert them to model inputs."""
        representation = representation_for_model(self.model_type)

        data = load_processed_dataset(
            dataset_name=self.dataset_name,
            representation=representation,
            processed_dir=self.processed_dir,
        )

        self.metadata = data["metadata"]
        self.cardinalities = self.metadata["cardinalities"]

        X_train_raw, y_train = dataframe_to_tensors(
            data["train_df"],
            device="cpu",
        )
        X_val_raw, y_val = dataframe_to_tensors(data["val_df"], device="cpu")
        X_test_raw, y_test = dataframe_to_tensors(
            data["test_df"],
            device="cpu",
        )

        if self.model_type in ONE_HOT_MODELS:
            categorical_cardinalities = self.metadata[
                "categorical_cardinalities"
            ]
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
        elif self.model_type in TENSOR_MODELS:
            self.X_train = X_train_raw.long()
            self.X_val = X_val_raw.long()
            self.X_test = X_test_raw.long()

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(self.metadata["target_mapping"])

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        dataset = TensorDataset(self.X_train, self.y_train)
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

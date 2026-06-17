# %% (rf is not included in here)
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data_pipeline.load_processed2 import (
    dataframe_to_tensors,
    load_processed_dataset,
)
from src.data_pipeline.encoding import one_hot_encode_features


ONE_HOT_MODELS = {"lr", "mlp"}
TENSOR_MODELS = {"cpd", "mba", "tt", "tr"}

# %% Sets the representation for each models depending on whether they are on hot encoded or tensor models

def representation_for_model(model_type):
    if model_type in ONE_HOT_MODELS:
        return "baseline"
    if model_type in TENSOR_MODELS:
        return "tensor"
    raise ValueError(f"Unsupported model type: {model_type}")

# %%

class TabularDataModule(L.LightningDataModule): # a lightning class that handles all dasa loading and prep.

    def __init__(self, dataset_name, mode, batch_size=256, model_type="lr",
                 processed_dir="src/data_pipeline/data/processed",
                 num_workers=0, seed=None, fold=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.fold = fold
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

    def setup(self, stage=None):
        representation = representation_for_model(self.model_type)

        data = load_processed_dataset(
            dataset_name=self.dataset_name,
            representation=representation,
            mode=self.mode,
            processed_dir=self.processed_dir,
            fold=self.fold,
        )

        self.metadata = data["metadata"]
        self.cardinalities = self.metadata["cardinalities"]

        if self.mode == "tuning": # handles tuning
            train_df = data["train_df"]
            val_df = data["val_df"]
            test_df = data["val_df"]  # tuning_val is used for both early stopping and final eval

        else:
            fold_test_df = data["val_df"]  # the held-out fold

            train_indices, val_indices = train_test_split(
                range(len(data["train_df"])),
                test_size=0.2,
                random_state=self.seed, # None would've been a random seed each time
            )
            train_df = data["train_df"].iloc[train_indices].reset_index(drop=True)
            val_df = data["train_df"].iloc[val_indices].reset_index(drop=True)
            test_df = fold_test_df

        X_train_raw, self.y_train = dataframe_to_tensors(train_df)
        X_val_raw, self.y_val = dataframe_to_tensors(val_df)
        X_test_raw, self.y_test = dataframe_to_tensors(test_df)

        if self.model_type in ONE_HOT_MODELS:
            categorical_cardinalities = self.metadata["categorical_cardinalities"]
            num_numerical_features = len(self.metadata["numerical_columns"])

            self.X_train = one_hot_encode_features(X_train_raw, categorical_cardinalities=categorical_cardinalities, num_numerical_features=num_numerical_features)
            self.X_val = one_hot_encode_features(X_val_raw, categorical_cardinalities=categorical_cardinalities, num_numerical_features=num_numerical_features)
            self.X_test = one_hot_encode_features(X_test_raw, categorical_cardinalities=categorical_cardinalities, num_numerical_features=num_numerical_features)

        elif self.model_type in TENSOR_MODELS:
            self.X_train = X_train_raw.long()
            self.X_val = X_val_raw.long()
            self.X_test = X_test_raw.long()

        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(self.metadata["target_mapping"])

    def train_dataloader(self): # wraps X_train and y_train in a TensorDataset (pairs each row of features with its label) and then returns a DataLoader that serves shuffled batches during training
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

    def val_dataloader(self): # same idea as above but for validation data
        dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self): # same as above but for test set
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
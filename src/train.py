from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_processed import load_processed_dataset, dataframe_to_tensors
from src.models.mlp import MLPClassifier
from src.utils.encoding import one_hot_encode_features
from src.models.logistic_regression import LogisticRegression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of processed dataset to load"
    )
    parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["lr", "mlp"],
    help="Model to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda", "auto"],
        help="Device to use for training (default: cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested, but it is not available on this machine.")
        return torch.device("mps")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested, but it is not available on this machine.")
        return torch.device("cuda")

    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_name}")


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_examples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.detach() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum()
            total_examples += batch_size

    avg_loss = (total_loss / total_examples).item()
    accuracy = (total_correct.float() / total_examples).item()
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset_name = args.dataset

    print(f"Training {args.model} on dataset: {dataset_name} (device={device.type})")

    data = load_processed_dataset(dataset_name=dataset_name, processed_dir="data/processed")
    metadata = data["metadata"]
    cardinalities = metadata["cardinalities"]

    X_train_int, y_train = dataframe_to_tensors(data["train_df"], device="cpu")
    X_val_int, y_val = dataframe_to_tensors(data["val_df"], device="cpu")
    X_test_int, y_test = dataframe_to_tensors(data["test_df"], device="cpu")

    X_train = one_hot_encode_features(X_train_int, cardinalities)
    X_val = one_hot_encode_features(X_val_int, cardinalities)
    X_test = one_hot_encode_features(X_test_int, cardinalities)

    # Move full splits once to avoid per-batch transfer overhead.
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = make_loader(X_train, y_train, batch_size=256, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=256, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(metadata["target_mapping"])

    if args.model == "lr":
        model = LogisticRegression(
            input_dim=input_dim,
            num_classes=num_classes,
        ).to(device)
    elif args.model == "mlp":
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = args.epochs

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = torch.zeros((), device=device)
        total_train_correct = torch.zeros((), device=device, dtype=torch.long)
        total_train_examples = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            total_train_loss += loss.detach() * batch_size
            preds = logits.argmax(dim=1)
            total_train_correct += (preds == y_batch).sum()
            total_train_examples += batch_size

        train_loss = (total_train_loss / total_train_examples).item()
        train_acc = (total_train_correct.float() / total_train_examples).item()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
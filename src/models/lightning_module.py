from __future__ import annotations

import lightning as L
import torch
from torch import nn


class TabularClassifierModule(L.LightningModule):
    """Lightning wrapper for tabular classifiers."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self._epoch_predictions: dict[str, list[torch.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self._epoch_targets: dict[str, list[torch.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _balanced_accuracy(
        preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        recalls = []
        for class_idx in range(num_classes):
            mask = targets == class_idx
            if mask.any():
                recalls.append((preds[mask] == targets[mask]).float().mean())

        if not recalls:
            return torch.tensor(0.0, device=targets.device)

        return torch.stack(recalls).mean()

    def _store_epoch_outputs(
        self,
        stage: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._epoch_predictions[stage].append(preds.detach().cpu())
        self._epoch_targets[stage].append(targets.detach().cpu())

    def _log_epoch_balanced_accuracy(self, stage: str) -> None:
        preds = self._epoch_predictions[stage]
        targets = self._epoch_targets[stage]
        if not preds or not targets:
            return

        epoch_preds = torch.cat(preds)
        epoch_targets = torch.cat(targets)
        num_classes = int(epoch_targets.max().item()) + 1
        balanced_acc = self._balanced_accuracy(
            preds=epoch_preds,
            targets=epoch_targets,
            num_classes=num_classes,
        )
        self.log(f"{stage}_balanced_acc", balanced_acc, prog_bar=False)
        preds.clear()
        targets.clear()

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and accuracy for one batch."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self._store_epoch_outputs(stage=stage, preds=preds, targets=y)
        return loss, acc

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, acc = self._shared_step(batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_epoch_balanced_accuracy(stage="train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, acc = self._shared_step(batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_balanced_accuracy(stage="val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, acc = self._shared_step(batch, stage="test")
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_balanced_accuracy(stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

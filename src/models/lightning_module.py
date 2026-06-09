from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch import nn


class TabularClassifierModule(L.LightningModule):
    """Lightning wrapper for tabular classifiers."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
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
    def _classification_metrics(
        preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> dict[str, torch.Tensor]:
        recalls = []
        precisions = []
        f1_scores = []
        supports = []

        for class_idx in range(num_classes):
            mask = targets == class_idx
            pred_mask = preds == class_idx
            support = mask.sum()
            if mask.any():
                true_positive = (pred_mask & mask).sum().float()
                recall = true_positive / support.float()
                predicted_positive = pred_mask.sum()
                precision = (
                    true_positive / predicted_positive.float()
                    if predicted_positive > 0
                    else torch.tensor(0.0, device=targets.device)
                )
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else torch.tensor(0.0, device=targets.device)
                )

                recalls.append(recall)
                precisions.append(precision)
                f1_scores.append(f1)
                supports.append(support.float())

        if not recalls:
            zero = torch.tensor(0.0, device=targets.device)
            return {
                "balanced_acc": zero,
                "macro_precision": zero,
                "macro_recall": zero,
                "macro_f1": zero,
                "weighted_f1": zero,
            }

        recall_tensor = torch.stack(recalls)
        precision_tensor = torch.stack(precisions)
        f1_tensor = torch.stack(f1_scores)
        support_tensor = torch.stack(supports)

        return {
            "balanced_acc": recall_tensor.mean(),
            "macro_precision": precision_tensor.mean(),
            "macro_recall": recall_tensor.mean(),
            "macro_f1": f1_tensor.mean(),
            "weighted_f1": (
                f1_tensor * support_tensor / support_tensor.sum()
            ).sum(),
        }

    @staticmethod
    def _confusion_matrix(
        preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.long,
            device=targets.device,
        )
        for target, pred in zip(targets, preds, strict=False):
            matrix[target.long(), pred.long()] += 1
        return matrix

    def _store_epoch_outputs(
        self,
        stage: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._epoch_predictions[stage].append(preds.detach().cpu())
        self._epoch_targets[stage].append(targets.detach().cpu())

    def _log_epoch_classification_metrics(self, stage: str) -> None:
        preds = self._epoch_predictions[stage]
        targets = self._epoch_targets[stage]
        if not preds or not targets:
            return

        epoch_preds = torch.cat(preds)
        epoch_targets = torch.cat(targets)
        num_classes = self.num_classes or int(epoch_targets.max().item()) + 1
        metrics = self._classification_metrics(
            preds=epoch_preds,
            targets=epoch_targets,
            num_classes=num_classes,
        )
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, prog_bar=False)
        if stage == "test":
            self._write_test_confusion_matrix(
                self._confusion_matrix(
                    preds=epoch_preds,
                    targets=epoch_targets,
                    num_classes=num_classes,
                )
            )
        preds.clear()
        targets.clear()

    def _write_test_confusion_matrix(self, matrix: torch.Tensor) -> None:
        if self.logger is None or not hasattr(self.logger, "log_dir"):
            return

        path = Path(self.logger.log_dir) / "test_confusion_matrix.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["true_class," + ",".join(
            f"pred_{class_idx}" for class_idx in range(matrix.shape[1])
        )]
        for class_idx, row in enumerate(matrix.cpu().tolist()):
            lines.append(
                f"{class_idx}," + ",".join(str(value) for value in row)
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_epoch_classification_metrics(stage="train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, acc = self._shared_step(batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_classification_metrics(stage="val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, acc = self._shared_step(batch, stage="test")
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_classification_metrics(stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Et simpelt feed-forward MLP til klassifikation.

    Består af to skjulte lag med ReLU aktivering og dropout.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
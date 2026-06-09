from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int, # number of input features
        hidden_dim: int = 128, # number of hidden units in each layer
        num_classes: int = 2, # number of output classes
        dropout: float = 0.1, # dropout rate for regularization
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

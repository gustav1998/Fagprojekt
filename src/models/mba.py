from __future__ import annotations

from itertools import combinations
from math import prod

import torch
from torch import nn


class MBAClassifier(nn.Module):
    """Supervised many-body classifier for discrete inputs."""

    def __init__(
        self,
        feature_dims: list[int], # Number of categories for each feature
        interaction_order: int, # Maximum order of interactions to consider
        num_classes: int, # Number of output classes
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("MBAClassifier needs at least one feature.")
        if interaction_order < 1:
            raise ValueError("interaction_order must be at least 1.")

        self.feature_dims = feature_dims
        self.interaction_order = min(interaction_order, len(feature_dims))
        self.num_classes = num_classes

        interactions: list[tuple[int, ...]] = [] # List of feature index combinations for interactions
        strides: list[torch.Tensor] = [] # List of stride tensors for each interaction to compute flat indices
        tables: list[nn.Parameter] = [] # List of parameter tensors for each interaction, storing the weights for each class and interaction combination

        for order in range(1, self.interaction_order + 1):
            for interaction in combinations(range(len(feature_dims)), order):
                dims = [feature_dims[index] for index in interaction]
                interactions.append(interaction)
                strides.append(self._make_strides(dims))
                tables.append(
                    nn.Parameter(
                        torch.empty(num_classes, prod(dims))
                    )
                )

        self.interactions = interactions
        self.interaction_tables = nn.ParameterList(tables)
        self.class_bias = nn.Parameter(torch.empty(num_classes))

        for idx, stride in enumerate(strides):
            self.register_buffer(f"_stride_{idx}", stride, persistent=False)

        self.reset_parameters()

    @staticmethod
    def _make_strides(dims: list[int]) -> torch.Tensor:
        strides: list[int] = []
        current = 1
        for dim in reversed(dims):
            strides.append(current)
            current *= dim
        return torch.tensor(list(reversed(strides)), dtype=torch.long)

    def reset_parameters(self) -> None:
        for table in self.interaction_tables:
            nn.init.normal_(table, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_dims):
            expected = len(self.feature_dims)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )

        logits = self.class_bias.unsqueeze(0).expand(x.size(0), -1)

        for idx, (interaction, table) in enumerate(
            zip(self.interactions, self.interaction_tables)
        ):
            values = x[:, interaction].long()
            strides = getattr(self, f"_stride_{idx}").to(x.device)
            flat_index = (values * strides).sum(dim=1)
            logits = logits + table[:, flat_index].T

        return logits

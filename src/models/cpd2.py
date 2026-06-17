from __future__ import annotations

import torch
from torch import nn

class ClassParafacClassifier(nn.Module):
    """
    Class-specific PARAFAC classifier for discrete inputs.

    For class c:

        z_c(x) = b_c + sum_r g[c, r] * product_d A_d[c, x_d, r]

    Each class has its own feature factors.
    """

    def __init__(
        self,
        feature_dims: list[int],
        rank: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("ClassParafacClassifier needs at least one feature")
        if rank < 1:
            raise ValueError("Rank must be at least 1")
        
        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        self.feature_factors = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(num_classes, feature_dim, rank)
                )
                for feature_dim in feature_dims
            ]
        )

        self.class_weights = nn.Parameter(torch.empty(num_classes, rank))
        self.class_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for factor in self.feature_factors:
            nn.init.uniform_(factor, a=0.9, b=1.1)

        nn.init.uniform_(self.class_weights, a=0.9, b=1.1)
        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_dims):
            expected = len(self.feature_dims)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )
        
        rank_components = torch.ones(
            x.size(0),
            self.num_classes,
            self.rank,
            device=x.device,
        )

        for feature_index, factor in enumerate(self.feature_factors):
            feature_values = x[:, feature_index].long()
            selected_factor_rows = factor[:, feature_values, :].permute(1, 0, 2)
            rank_components = rank_components * selected_factor_rows

        weighted_components = rank_components * self.class_weights.unsqueeze(0)
        logits = weighted_components.sum(dim=2) + self.class_bias

        return logits
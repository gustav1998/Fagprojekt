from __future__ import annotations

import torch
from torch import nn


class CPDClassifier(nn.Module):
    """Supervised CPD classifier for discrete inputs.

    For an input x = (x_1, ..., x_D), the class-c logit is

        z_c(x) = sum_r lambda[c, r] * prod_d A_d[x_d, r] + bias[c]

    This matches the supervised CPD score used in the report: the CPD tensor
    directly parameterizes class logits, and the Lightning wrapper trains those
    logits with cross-entropy.
    """
    def __init__(
        self,
        feature_dims: list[int],
        rank: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        # A_d in the report: one factor matrix per feature/mode.
        self.feature_factors = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, rank)) for dim in feature_dims]
        )

        # lambda[c, r] and bias[c]: class-specific CPD component weights.
        self.class_weights = nn.Parameter(torch.empty(num_classes, rank))
        self.class_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for factor in self.feature_factors:
            nn.init.normal_(factor, mean=1.0, std=0.01)

        nn.init.normal_(self.class_weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_factors):
            expected = len(self.feature_factors)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )

        rank_components = torch.ones(
            x.size(0),
            self.rank,
            device=x.device,
        )

        for j, factor in enumerate(self.feature_factors):
            feature_values = x[:, j].long()
            rank_components = rank_components * factor[feature_values]

        return rank_components @ self.class_weights.T + self.class_bias

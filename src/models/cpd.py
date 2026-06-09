from __future__ import annotations

import torch
from torch import nn


class CPDClassifier(nn.Module):
    def __init__(
        self,
        feature_dims: list[int], # list of the number of unique values for each categorical feature
        rank: int, # rank of CPD factorization
        num_classes: int, # number of output classes
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("CPDClassifier needs at least one feature.")

        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        # creates one empty factor matrix per input feature:
        self.feature_factors = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, rank)) for dim in feature_dims]
        )

        self.class_weights = nn.Parameter(torch.empty(num_classes, rank)) # sets the class weights
        self.class_bias = nn.Parameter(torch.empty(num_classes)) # sets the class bias

        self.reset_parameters()

    # resets the parameters of the model before training
    def reset_parameters(self) -> None:
        for factor in self.feature_factors:
            nn.init.normal_(factor, mean=1.0, std=0.01)

        nn.init.normal_(self.class_weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_bias)

    # forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_factors):
            expected = len(self.feature_factors)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )

        # compute the rank components
        rank_components = torch.ones(
            x.size(0),
            self.rank,
            device=x.device,
        )
        for j, factor in enumerate(self.feature_factors):
            feature_values = x[:, j].long()
            rank_components = rank_components * factor[feature_values]

        # compute and return the class logits
        return rank_components @ self.class_weights.T + self.class_bias

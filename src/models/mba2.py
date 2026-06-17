from __future__ import annotations

from itertools import combinations

import torch
from torch import nn


class MBAClassifier(nn.Module):
    """MBA classifier with single-feature and pairwise interactions. If needed, can be extended to higher-order interactions, but this is not implemented yet."""

    def __init__(
        self,
        feature_dims: list[int],
        interaction_order: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if interaction_order not in (1, 2):
            raise ValueError("Model currently only supports interaction_order 1 or 2.")

        self.feature_dims = feature_dims
        self.num_classes = num_classes

        self.single_tables = nn.ParameterList(
            [
                nn.Parameter(torch.empty(num_classes, dim))
                for dim in feature_dims
            ]
        )

        self.pairs = list(combinations(range(len(feature_dims)), 2))

        self.pair_tables = nn.ParameterList()
        if interaction_order == 2:
            self.pair_tables = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(
                            num_classes,
                            feature_dims[i],
                            feature_dims[j],
                        )
                    )
                    for i, j in self.pairs
                ]
            )



        self.class_bias = nn.Parameter(torch.empty(num_classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in self.single_tables:
            nn.init.normal_(table, mean=0.0, std=0.01)

        for table in self.pair_tables:
            nn.init.normal_(table, mean=0.0, std=0.01)

        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.class_bias.unsqueeze(0).expand(x.size(0), -1)

        for feature_index, table in enumerate(self.single_tables):
            values = x[:, feature_index].long()
            logits = logits + table[:, values].T

        for (i, j), table in zip(self.pairs, self.pair_tables):
            first_values = x[:, i].long()
            second_values = x[:, j].long()
            logits = logits + table[:, first_values, second_values].T

        return logits
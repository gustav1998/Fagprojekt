from __future__ import annotations

import torch
from torch import nn


class TRClassifier(nn.Module):
    """Supervised tensor-ring classifier for discrete inputs."""

    def __init__(
        self,
        feature_dims: list[int],
        rank: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("TRClassifier needs at least one feature.")

        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        self.feature_cores = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, rank, rank)) for dim in feature_dims]
        )
        self.class_matrices = nn.Parameter(
            torch.empty(num_classes, rank, rank)
        )
        self.class_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        eye = torch.eye(self.rank)
        for core in self.feature_cores:
            with torch.no_grad():
                core.copy_(
                    eye.expand(core.shape[0], self.rank, self.rank)
                    + 0.01 * torch.randn_like(core)
                )

        nn.init.normal_(self.class_matrices, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_cores):
            expected = len(self.feature_cores)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )

        ring_state = self.feature_cores[0][x[:, 0].long()]

        for j, core in enumerate(self.feature_cores[1:], start=1):
            matrices = core[x[:, j].long()]
            ring_state = torch.bmm(ring_state, matrices)

        logits = torch.einsum("bij,cji->bc", ring_state, self.class_matrices)
        return logits + self.class_bias

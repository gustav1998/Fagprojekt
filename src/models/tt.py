from __future__ import annotations

import torch
from torch import nn


class TTClassifier(nn.Module):
    """Supervised tensor-train classifier for discrete inputs."""

    def __init__(
        self,
        feature_dims: list[int],
        rank: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("TTClassifier needs at least one feature.")

        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        cores: list[nn.Parameter] = [
            nn.Parameter(torch.empty(feature_dims[0], 1, rank))
        ]
        cores.extend(
            nn.Parameter(torch.empty(dim, rank, rank))
            for dim in feature_dims[1:]
        )
        self.feature_cores = nn.ParameterList(cores)

        self.class_weights = nn.Parameter(torch.empty(rank, num_classes))
        self.class_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_cores[0], mean=1.0, std=0.01)

        eye = torch.eye(self.rank)
        for core in self.feature_cores[1:]:
            with torch.no_grad():
                core.copy_(
                    eye.expand(core.shape[0], self.rank, self.rank)
                    + 0.01 * torch.randn_like(core)
                )

        nn.init.normal_(self.class_weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_cores):
            expected = len(self.feature_cores)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )

        state = self.feature_cores[0][x[:, 0].long()].squeeze(1)

        for j, core in enumerate(self.feature_cores[1:], start=1):
            matrices = core[x[:, j].long()]
            state = torch.bmm(state.unsqueeze(1), matrices).squeeze(1)

        return state @ self.class_weights + self.class_bias

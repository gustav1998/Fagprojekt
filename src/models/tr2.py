from __future__ import annotations

import torch
from torch import nn


class ClassTRClassifier(nn.Module):
    """
    Class-specific Tensor Ring classifier for discrete inputs.

    For class c:

        z_c(x) = b_c + trace(A_0^(c)[x_0] A_1^(c)[x_1] ... A_{D-1}^(c)[x_{D-1}])

    Each selected core slice is an R x R matrix.
    """

    def __init__(
        self,
        feature_dims: list[int],
        rank: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("ClassTRClassifier needs at least one feature.")
        if rank < 1:
            raise ValueError("rank must be at least 1.")

        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        self.feature_cores = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        num_classes,
                        feature_dim,
                        rank,
                        rank,
                    )
                )
                for feature_dim in feature_dims
            ]
        )

        self.class_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        identity = torch.eye(self.rank)

        for core in self.feature_cores:
            with torch.no_grad():
                core.copy_(
                    identity.expand(
                        self.num_classes,
                        core.shape[1],
                        self.rank,
                        self.rank,
                    )
                    + 0.01 * torch.randn_like(core)
                )
            
        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_dims):
            expected = len(self.feature_dims)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )
        
        first_core = self.feature_cores[0]
        first_values = x[:, 0].long()

        ring_state = first_core[:, first_values, :, :].permute(1, 0, 2, 3)

        for feature_index, core in enumerate(self.feature_cores[1:], start=1):
            feature_values = x[:, feature_index].long()
            selected_core = core[:, feature_values, :, :].permute(1, 0, 2, 3)

            batch_size = ring_state.size(0)

            ring_state_as_matrix = ring_state.reshape(
                batch_size * self.num_classes,
                self.rank,
                self.rank,
            )

            selected_core_as_matrix = selected_core.reshape(
                batch_size * self.num_classes,
                self.rank,
                self.rank,
            )

            multiplied_state = torch.bmm(
                ring_state_as_matrix,
                selected_core_as_matrix,
            )

            ring_state = multiplied_state.reshape(
                batch_size,
                self.num_classes,
                self.rank,
                self.rank,
            )

        logits = torch.diagonal(
            ring_state,
            dim1=2,
            dim2=3,
        ).sum(dim=2)

        logits = logits + self.class_bias

        return logits
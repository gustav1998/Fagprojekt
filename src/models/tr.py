from __future__ import annotations

import torch
from torch import nn


class TRClassifier(nn.Module):
    """Supervised tensor-ring classifier for discrete inputs."""

    def __init__(
        self,
        feature_dims: list[int], # list of the number of unique values for each categorical feature
        rank: int, # rank of TR factorization
        num_classes: int, # number of output classes
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("TRClassifier needs at least one feature.")

        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        # creates one empty factor matrix per input feature:
        self.feature_cores = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, rank, rank)) for dim in feature_dims]
        )

        # sets the class weights and bias for the final linear layer that maps the contracted TR representation to class logits
        self.class_matrices = nn.Parameter(
            torch.empty(num_classes, rank, rank)
        ) #
        self.class_bias = nn.Parameter(torch.empty(num_classes)) 

        self.reset_parameters()

    # resets the parameters of the model before training
    def reset_parameters(self) -> None:

        # initializes the remaining cores close to identity matrices 
        eye = torch.eye(self.rank)
        for core in self.feature_cores:
            with torch.no_grad():
                core.copy_(
                    eye.expand(core.shape[0], self.rank, self.rank)
                    + 0.01 * torch.randn_like(core)
                )

        nn.init.normal_(self.class_matrices, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_bias)

    # forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.feature_cores):
            expected = len(self.feature_cores)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )

        # computes the TR contraction for each sample in the batch, resulting in a final state vector of size (batch_size, rank, rank)
        ring_state = self.feature_cores[0][x[:, 0].long()]

        for j, core in enumerate(self.feature_cores[1:], start=1):
            matrices = core[x[:, j].long()]
            ring_state = torch.bmm(ring_state, matrices)

        # compute and return the class logits by contracting the final ring state with the class weight matrices and adding the class bias
        logits = torch.einsum("bij,cji->bc", ring_state, self.class_matrices)
        return logits + self.class_bias

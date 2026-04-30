from __future__ import annotations

import torch
from torch import nn


class CPDClassifier(nn.Module):
    """CPD-baseret klassifikationsmodel.

    Hver kategorisk feature har en embedding af størrelse `rank`. Vi multiplicerer
    embeddings elementvist (CPD-lignende) for at få en samlet repræsentation,
    som derefter ligneært projiceres til `num_classes` logits.
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

        # En embedding per kategorisk feature (dim -> rank)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, rank) for dim in feature_dims]
        )

        # Lineært output fra rank-dimension til klasse-logits
        self.output = nn.Linear(rank, num_classes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=1.0, std=0.01)

        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Start med en vektor af 1'ere i rank-dimensionen
        z = torch.ones(
            x.size(0),
            self.rank,
            device=x.device,
        )

        # For hver feature: hent embedding per sample og multiplicer elementvist
        for j, embedding in enumerate(self.embeddings):
            feature_values = x[:, j].long()
            z = z * embedding(feature_values)

        logits = self.output(z)
        return logits
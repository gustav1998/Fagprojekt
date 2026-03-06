from __future__ import annotations

import torch
import torch.nn.functional as F


def one_hot_encode_features(
    X: torch.Tensor,
    cardinalities: list[int],
) -> torch.Tensor:
    """
    Convert integer-coded categorical features into one-hot encoded features.

    Args:
        X: Tensor of shape (batch_size, num_features), dtype long
        cardinalities: List of number of categories for each feature

    Returns:
        Tensor of shape (batch_size, sum(cardinalities)), dtype float
    """
    one_hot_parts = []

    for feature_idx, cardinality in enumerate(cardinalities):
        feature_values = X[:, feature_idx]
        one_hot = F.one_hot(feature_values, num_classes=cardinality).float()
        one_hot_parts.append(one_hot)

    return torch.cat(one_hot_parts, dim=1)
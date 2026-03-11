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
    num_categorical = len(cardinalities)
    num_total_features = X.shape[1]

    if num_categorical > num_total_features:
        raise ValueError(
            "Cardinalities length cannot exceed number of features. "
            f"Got {num_categorical} cardinalities for {num_total_features} features."
        )

    one_hot_parts = []

    for feature_idx, cardinality in enumerate(cardinalities):
        feature_values = X[:, feature_idx].long()
        one_hot = F.one_hot(feature_values, num_classes=cardinality).float()
        one_hot_parts.append(one_hot)

    if num_categorical < num_total_features:
        numerical_part = X[:, num_categorical:].float()
        one_hot_parts.append(numerical_part)

    if not one_hot_parts:
        return torch.empty((X.shape[0], 0), dtype=torch.float32, device=X.device)

    return torch.cat(one_hot_parts, dim=1)
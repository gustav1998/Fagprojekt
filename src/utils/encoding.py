from __future__ import annotations

import torch
import torch.nn.functional as F


def one_hot_encode_features(
    X: torch.Tensor,
    categorical_cardinalities: list[int],
    num_numerical_features: int = 0,
) -> torch.Tensor:
    """
    One-hot encode the categorical part of X and append the numerical part unchanged.

    Expected column order in X:
    [categorical features..., numerical features...]

    Args:
        X: Tensor of shape (batch_size, num_features)
        categorical_cardinalities: Cardinalities for categorical columns only
        num_numerical_features: Number of numerical columns at the end of X

    Returns:
        Tensor of shape (batch_size, sum(categorical_cardinalities) + num_numerical_features)
    """
    num_categorical = len(categorical_cardinalities)
    expected_total = num_categorical + num_numerical_features

    if X.shape[1] != expected_total:
        raise ValueError(
            f"Expected {expected_total} features "
            f"({num_categorical} categorical + {num_numerical_features} numerical), "
            f"but got {X.shape[1]}."
        )

    parts: list[torch.Tensor] = []

    for feature_idx, cardinality in enumerate(categorical_cardinalities):
        feature_values = X[:, feature_idx].long()
        one_hot = F.one_hot(feature_values, num_classes=cardinality).float()
        parts.append(one_hot)

    if num_numerical_features > 0:
        numerical_part = X[:, num_categorical:].float()
        parts.append(numerical_part)

    if not parts:
        return torch.empty((X.shape[0], 0), dtype=torch.float32, device=X.device)

    return torch.cat(parts, dim=1)
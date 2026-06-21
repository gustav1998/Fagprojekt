from __future__ import annotations

import torch
from torch import nn

class ClassTTClassifier(nn.Module):
    """
    Class-specific TT classifier for discrete inputs.

    For class c:

        z_c(x) = b_c + A_0[c, x_0] A_1[c, x_1] ... A_{D-1}[c, x_{D-1}]

    The first and last TT ranks are 1, so the product is scalar.
    """

    def __init__(
        self,
        feature_dims: list[int],
        rank: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("ClassTTClassifier needs at least one feature")
        if rank < 1:
            raise ValueError("Rank must be at least 1")
        
        self.feature_dims = feature_dims
        self.rank = rank
        self.num_classes = num_classes

        cores: list[nn.Parameter] = []

        num_features = len(feature_dims)

        for feature_index, feature_dim in enumerate(feature_dims):
            if feature_index == 0:
                left_rank = 1
            else:
                left_rank = rank

            if feature_index == num_features - 1:
                right_rank = 1
            else:
                right_rank = rank

            core = nn.Parameter(
                torch.empty(
                    num_classes,
                    feature_dim,
                    left_rank,
                    right_rank,
                )
            )
            cores.append(core)

        self.feature_cores = nn.ParameterList(cores)

        self.class_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for core in self.feature_cores:
            nn.init.normal_(core, mean=0.0, std=0.01)

        nn.init.zeros_(self.class_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x contains a batch of discrete feature values.
        # Shape: batch_size x number_of_features.
        if x.shape[1] != len(self.feature_dims):
            expected = len(self.feature_dims)
            raise ValueError(
                f"Expected {expected} features, got {x.shape[1]}."
            )
        
        # Start with the first TT core, A_0.
        # Shape: num_classes x feature_dim_0 x 1 x rank.
        first_core = self.feature_cores[0]

        # Select the observed value of feature 0 for every sample.
        # Shape: batch_size.
        first_values = x[:, 0].long()

        # For each sample and class, select the matrix A_0[c, x_0].
        #
        # Before permute:
        # first_core[:, first_values, :, :] has shape
        # num_classes x batch_size x 1 x rank.
        #
        # After permute:
        # state has shape
        # batch_size x num_classes x 1 x rank.
        state = first_core[:, first_values, :, :].permute(1, 0, 2, 3)

        # Multiply in the remaining TT cores:
        # A_1[c, x_1], A_2[c, x_2], ..., A_{D-1}[c, x_{D-1}].
        for feature_index, core in enumerate(self.feature_cores[1:], start=1):
            # Select the observed value of the current feature for every sample.
            # Shape: batch_size.
            feature_values = x[:, feature_index].long()

            # For each sample and class, select the matrix A_i[c, x_i].
            #
            # Before permute:
            # core[:, feature_values, :, :] has shape
            # num_classes x batch_size x left_rank x right_rank.
            #
            # After permute:
            # selected_core has shape
            # batch_size x num_classes x left_rank x right_rank.
            selected_core = core[:, feature_values, :, :].permute(1, 0, 2, 3)

            # torch.bmm does batched matrix multiplication, but it expects
            # tensors with shape: batch x rows x columns.
            #
            # Our tensors also have a class dimension, so we temporarily merge
            # batch_size and num_classes into one dimension.
            batch_size = state.size(0)
            state_as_matrix = state.reshape(
                batch_size * self.num_classes,
                state.size(2),
                state.size(3),
            )

            # Shape:
            # batch_size * num_classes x middle_rank x right_rank.
            selected_core_as_matrix = selected_core.reshape(
                batch_size * self.num_classes,
                selected_core.size(2),
                selected_core.size(3),
            )

            # Multiply the previous product by the current selected core.
            # Mathematically, this updates:
            # state = A_0[c, x_0] A_1[c, x_1] ... A_i[c, x_i].
            multiplied_state = torch.bmm(
                state_as_matrix,
                selected_core_as_matrix
            )

            # Reshape back to:
            # batch_size x num_classes x left_rank x right_rank.
            state = multiplied_state.reshape(
                batch_size,
                self.num_classes,
                multiplied_state.size(1),
                multiplied_state.size(2),
            )
        
        # After all TT cores have been multiplied, the first and last TT ranks
        # make the final shape: batch_size x num_classes x 1 x 1.
        # Removing those last two dimensions gives one logit per class.
        logits = state.squeeze(-1).squeeze(-1)

        # Add one learned bias per class.
        # class_bias shape: num_classes.
        # logits shape: batch_size x num_classes.
        logits = logits + self.class_bias

        return logits

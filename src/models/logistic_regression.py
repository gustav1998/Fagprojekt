import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_classes: int = 2,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
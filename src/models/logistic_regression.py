import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(
            self,
            input_dim: int, # number of input features
            num_classes: int = 2, # number of output classes
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x): # forward pass 
        return self.linear(x)
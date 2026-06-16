import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                input_dim, 
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )
    
    def forward(self, x):

        return self.mlp(x)
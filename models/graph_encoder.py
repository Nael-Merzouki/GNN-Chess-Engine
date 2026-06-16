import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GINEConv, global_add_pool)


class GraphEncoder(nn.Module):

    def __init__(self, in_channels=8, hidden_dim=64):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINEConv(mlp1, edge_dim=3)

        self.conv2 = GINEConv(mlp2, edge_dim=3)
    
    def forward(self, x, edge_index, edge_attr, batch):

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)

        graph_embedding = global_add_pool(x, batch)

        return graph_embedding
import torch.nn as nn
from models.graph_encoder import GraphEncoder
from models.value_head import ValueHead


class ChessGNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = GraphEncoder(
            in_channels=8,
            hidden_dim=64,
            num_layers=3,
            heads=4
        )

        self.value_head = ValueHead(input_dim=64)
    
    def forward(self, data):

        embedding = self.encoder(
            data.x,
            data.edge_index,
            data.batch
        )

        value = self.value_head(embedding)

        return value
import torch
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, global_add_pool)


class GraphEncoder(torch.nn.Module):

    def __init__(self, in_channels=8, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)

        #self.convs = torch.nn.ModuleList()

        #self.convs.append(
            #GATConv(
                #in_channels=in_channels,
                #out_channels=hidden_dim,
                #heads=heads,
                #dropout=0.2
            #)
        #

        #for _ in range(num_layers - 1):

            #self.convs.append(
                #GATConv(
                    #in_channels=hidden_dim * heads,
                    #out_channels=hidden_dim,
                    #heads=heads,
                    #dropout=0.2
                #)
            #)
    
    def forward(self, x, edge_index, batch):
        
        #for conv in self.convs:

            #x = conv(x, edge_index)

            #x = F.relu(x)
        
        #graph_embedding = global_mean_pool(x, batch)

        #return graph_embedding

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        return global_add_pool(x, batch)
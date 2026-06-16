import torch

from graph_dataset import fen_to_graph


#fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

#g = fen_to_graph(fen)

#print(g)
#print(g.x.shape)
#print(g.edge_index.shape)

import torch


graphs = torch.load('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/graph_dataset.pt', weights_only=False)

#g = graphs[0]

#print(g)
#print(g.x.shape)
#print(g.edge_index.shape)
#print(g.y_value)
#print(g.policy)

ys = torch.stack(
    [g.y_value for g in graphs]
).squeeze()

print("min:", ys.min())
print("max:", ys.max())
print("mean:", ys.mean())
print("std:", ys.std())

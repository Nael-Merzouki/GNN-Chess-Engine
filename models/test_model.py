import torch
from torch_geometric.loader import DataLoader
from model import ChessGNN


graphs = torch.load('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/graph_dataset.pt', weights_only=False)

loader = DataLoader(graphs[:32], batch_size=8)

model = ChessGNN()

batch = next(iter(loader))

out = model(batch)

print(out)
print(out.shape)
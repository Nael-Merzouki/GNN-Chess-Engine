import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# =========================
# SPLITTING & LOADING
# =========================
graphs = torch.load('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/graph_dataset.pt', weights_only=False)
OUTPUT_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/experiments/value_model.pt'

n = len(graphs)

train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_graphs = graphs[:train_end]
val_graphs = graphs[train_end:val_end]
test_graphs = graphs[val_end:]

print(train_end)
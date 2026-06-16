from math import e

import torch
from torch_geometric.loader import DataLoader
from models.model import ChessGNN
from tqdm import tqdm
import copy


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

#print(type(graphs[0]).__name__)

train_loader = DataLoader(train_graphs,
                          batch_size=64,
                          shuffle=True)

val_loader = DataLoader(val_graphs,
                        batch_size=64,
                        shuffle=False)

# ===============
# CUDA
# ===============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print(device)

# ==================
# HYPERPARAMETERS
# ==================
model = ChessGNN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

criterion = torch.nn.MSELoss()

EPOCHS = 75

# ================
# TRAINING LOOP
# ================
train_losses = []
val_losses = []

best_val_loss = float('inf')
best_epoch = -1

patience = 10
min_delta = 0.001
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for batch in tqdm(train_loader):
        
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)

        target = batch.y_value

        loss = criterion(pred.squeeze(), target.squeeze())

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    train_losses.append(avg_train_loss)

    # Validation
    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(device)

            loss = criterion(model(batch).squeeze(), batch.y_value.squeeze())

            total_val_loss += loss.item()
    
        avg_val_loss = total_val_loss / len(val_loader)

        val_losses.append(avg_val_loss)
    
    # Implement early stopping
    if avg_val_loss < best_val_loss - min_delta:
        
        best_val_loss = avg_val_loss
        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(model.state_dict(), 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/experiments/best_value_model.pt')
    else:

        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:

        print(f'Early stopping at epoch {epoch+1}')

        break
    
    print(f'Epoch {epoch+1}: train loss = {avg_train_loss:.4f} | val loss = {avg_val_loss:.4f}')

# =================
# SAVING
# =================
torch.save(model.state_dict(), OUTPUT_PATH)

# ====================
# VALIDATION & TEST
# ====================
BEST_MODEL_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/experiments/best_value_model.pt'

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

print(f'Loaded best model from epoch {best_epoch+1}.')

val_loss = 0

with torch.no_grad():

    for batch in val_loader:

        batch = batch.to(device)

        pred = model(batch)

        loss = criterion(pred.squeeze(), batch.y_value.squeeze())

        val_loss += loss.item()

print(f'Validation Loss: {val_loss / len(val_loader)}')
print(f'Train Loss History: {train_losses}')
print(f'Validation Loss History: {val_losses}')
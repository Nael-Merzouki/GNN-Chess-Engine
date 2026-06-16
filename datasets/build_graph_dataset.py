from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from datasets.graph_dataset import fen_to_graph

# ===============
# CONFIG
# ===============
INPUT_PATH = Path('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/labeled_positions.parquet')
OUTPUT_PATH = Path('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/graph_dataset.pt')
MAX_ROWS = float('inf')

# =====================
# GRAPH CONVERSION
# =====================
def main():

    df = pd.read_parquet(INPUT_PATH)
    
    # df = df.head(MAX_ROWS)

    graphs = []

    for row in tqdm(df.itertuples(index=False), total=len(df)):

        graph = fen_to_graph(
            row.fen, 
            row.stockfish_eval,
            row.best_move
        )

        graphs.append(graph)

# =============
# SAVING
# =============
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(graphs, OUTPUT_PATH)

    print(f'Saved {len(graphs)} graphs to {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
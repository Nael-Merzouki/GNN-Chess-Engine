import chess
import chess.engine
import pandas as pd
from tqdm import tqdm


# =================
# CONFIG
# =================
INPUT_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/filtered_positions.parquet'
OUTPUT_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/labeled_positions.parquet'
ENGINE_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/stockfish/stockfish-windows-x86-64-avx2.exe'
MAX_ROWS = 100

# ==============
# LOADING
# ==============
df = pd.read_parquet(INPUT_PATH)

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

results = []
#row_count = 1

for _, row in tqdm(df.iterrows(), total=len(df)):

    #if row_count > MAX_ROWS:
    #    break

    board = chess.Board(row['fen'])

    info = engine.analyse(board, chess.engine.Limit(time=0.05))

    score = info['score'].relative

    if score.is_mate():
        eval_cp = 10000 if score.mate() > 0 else -10000
    else:
        eval_cp = score.score()
    
    best_move = info['pv'][0].uci() if 'pv' in info else None

    results.append({
        **row.to_dict(),
        'stockfish_eval': eval_cp,
        'best_move': best_move
    })

    #row_count += 1

engine.quit()

out = pd.DataFrame(results)
out.to_parquet(OUTPUT_PATH, index=False)

print(f'Done: {len(out)}')
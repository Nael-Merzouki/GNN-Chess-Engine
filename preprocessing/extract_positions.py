from pathlib import Path
import io
import zstandard as zstd
import pandas as pd
from tqdm import tqdm
import chess.pgn


print('OUTPUT:\n')

# =================
# CONFIG
# =================
PGN_PATH = Path('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/raw/lichess_db_standard_rated_2026-01.pgn.zst')
OUTPUT_PATH = Path('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/positions.parquet')
MAX_GAMES = 5000

# =================
# OPEN STREAM
# =================
def open_pgn_stream(path):
    compressed_file = open(path, 'rb')
    dctx = zstd.ZstdDecompressor()

    stream = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(stream, encoding='utf-8')

    return compressed_file, text_stream

# ==================
# EXTRACTION
# ==================
def extract_positions():
    data = []

    compressed_file, pgn = open_pgn_stream(PGN_PATH)

    try:
        game_count = 0

        while game_count < MAX_GAMES:
            game = chess.pgn.read_game(pgn)

            if game is None:
                continue
            
            if len(list(game.mainline_moves())) < 5:
                continue

            game_count += 1

            board = game.board()

            headers = game.headers
            white_elo = headers.get('WhiteElo', None)
            black_elo = headers.get('BlackElo', None)
            result = headers.get('Result', None)

            if (white_elo is None) or (black_elo is None):
                continue

            for ply, move in enumerate(game.mainline_moves(), start=1):
                
                fen = board.fen()

                data.append({
                    'fen': fen,
                    'move': move.uci(),
                    'white_elo': white_elo,
                    'black_elo': black_elo,
                    'result': result,
                    'ply': ply
                })

                board.push(move)
            
            if game_count % 100 == 0:
                print(f'Processed {game_count} games | rows: {len(data)}')
    
    finally:
        compressed_file.close()
    
    df = pd.DataFrame(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f'Saved dataset: {OUTPUT_PATH}')
    print(f'Total positions: {len(df)}')

if __name__ == '__main__':
    extract_positions()
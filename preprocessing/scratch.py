import sys
# print(sys.executable)

import chess
# print(chess.__version__)


print(f"OUPUT:\n")

from pathlib import Path
path = Path('data/raw')
print(path.exists())
print(list(path.iterdir()))

# board = chess.Board()

# print(f"{board}\n")

# board.push_san('e4')
# print(board.fen())
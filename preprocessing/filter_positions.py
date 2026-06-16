import pandas as pd


# ================
# CONFIG
# ================
INPUT_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/positions.parquet'
OUTPUT_PATH = 'C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/filtered_positions.parquet'
MIN_ELO = 2000
MIN_PLY = 8

# ===================
# FORMATTING
# ===================
df = pd.read_parquet(INPUT_PATH)

unfiltered_len = len(df)

# print(df.head(10))

df['white_elo'] = df['white_elo'].astype(int)
df['black_elo'] = df['black_elo'].astype(int)

df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2

# ==================
# FILTERING
# ==================
df = df[df['avg_elo'] >= MIN_ELO]
df = df[df['ply'] >= MIN_PLY]
df = df[df['ply'] <= 120]

# ==============
# SAVE
# ==============
df.to_parquet(OUTPUT_PATH, index=False)

print(f'Original: {unfiltered_len}')
print(f'Filtered: {len(df)}')
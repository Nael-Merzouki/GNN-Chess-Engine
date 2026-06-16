import pandas as pd


df = pd.read_parquet('C:/Users/mnael/OneDrive/Documents/Nael/Code/VisualStudio/projects/chess-ai/data/processed/labeled_positions.parquet')
print(df.head(5))
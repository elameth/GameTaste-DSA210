import pandas as pd

df = pd.read_csv("data/rawg_games_filtered_10k.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isna().sum())

"""
Objective 3: ML-Based Recommendations
- Trains a Random Forest Regressor on your rating history.
- Predicts ratings for all unplayed games in the 10k dataset.
- Outputs the top 20 games you are most likely to enjoy.
"""

import pandas as pd
import numpy as np
import re
import ast
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

# --- Configuration ---
# Adjust these paths if necessary
PROCESSED_DIR = Path("data/processed")
# If running from root, try these:
if not PROCESSED_DIR.exists():
    PROCESSED_DIR = Path(".")  # Current dir fallback

RATINGS_PATH = PROCESSED_DIR / "my_ratings_template.csv"
RAWG_PATH = PROCESSED_DIR / "rawg_games_filtered_10k.csv"  # Or use the _clean version if you made it



# --- Utilities ---

def normalize_name(s):
    """Normalize names to catch slight variations (e.g. 'Skyrim' vs 'The Skyrim')"""
    return re.sub(r"[^a-z0-9\s]+", " ", str(s).lower()).strip()


def parse_multilabel(x):
    """Parses tags/genres from string format to list."""
    if pd.isna(x): return []

    # Handle list-strings or pipe-separated strings
    if str(x).startswith("["):
        try:
            res = ast.literal_eval(x)
        except:
            res = []
    else:
        res = re.split(r"[|,;]", str(x))

    clean = []
    for t in res:
        t_clean = str(t).strip()
        if t_clean and t_clean.lower():
            clean.append(t_clean)
    return sorted(list(set(clean)))


# --- Main Logic ---

def main():
    print("Loading data...")
    # Robust load
    try:
        ratings = pd.read_csv(RATINGS_PATH)
        rawg = pd.read_csv(RAWG_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find files at {RATINGS_PATH} or {RAWG_PATH}")
        return

    # Normalize Names for Matching
    ratings["_norm"] = ratings["name"].apply(normalize_name)
    rawg["_norm"] = rawg["name"].apply(normalize_name)

    # Merge: Keep ALL rawg games (left join), add ratings where they exist
    merged = rawg.merge(ratings[["_norm", "my_rating_10"]], on="_norm", how="left")

    # Split into TRAINING (Played) and PREDICTION (Unplayed) sets
    train_df = merged[merged["my_rating_10"].notna()].copy()
    predict_df = merged[merged["my_rating_10"].isna()].copy()

    print(f"Training Model on: {len(train_df)} games")
    print(f"Generating Recs for: {len(predict_df)} unplayed games")

    # --- Feature Engineering ---

    print("Building features...")
    # Process Tags & Genres for ALL games together to ensure consistent columns
    all_genres = merged["genres"].apply(parse_multilabel)
    all_tags = merged["tags"].apply(parse_multilabel)

    labels = (all_genres.apply(lambda x: [f"GENRE_{i}" for i in x]) +
              all_tags.apply(lambda x: [f"TAG_{i}" for i in x])).tolist()

    # Create Binary Features (One-Hot Encoding)
    mlb = MultiLabelBinarizer()
    X_all_cat = pd.DataFrame(mlb.fit_transform(labels), index=merged.index, columns=mlb.classes_)

    # Add Numerical Features
    # 1. Year (Impute missing with median ~2010)
    X_all_cat["NUM_Year"] = pd.to_numeric(merged["released_year"], errors="coerce").fillna(2010)

    # 2. Metacritic (Impute missing with generic 70)
    if "metacritic" in merged.columns:
        X_all_cat["NUM_Metacritic"] = pd.to_numeric(merged["metacritic"], errors="coerce").fillna(70)

    # 3. Log(Ratings Count) - Helps prioritize popular games over obscure ones with 1 vote
    if "ratings_count" in merged.columns:
        X_all_cat["NUM_LogRatingsCount"] = np.log1p(pd.to_numeric(merged["ratings_count"], errors="coerce").fillna(0))

    # Force column names to strings (fixes sklearn error)
    X_all_cat.columns = X_all_cat.columns.astype(str)

    # --- Training ---

    # Extract Training Data
    X_train = X_all_cat.loc[train_df.index]
    y_train = train_df["my_rating_10"]

    # Drop columns that never appear in training (Model can't learn them)
    valid_cols = X_train.columns[X_train.sum() > 0]
    X_train = X_train[valid_cols]

    print(f"Training Random Forest on {X_train.shape[1]} features...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)

    # --- Prediction ---

    # Prepare Unplayed Data (Must have same columns as X_train)
    X_predict = X_all_cat.loc[predict_df.index]
    # Keep only valid columns, fill missing (features not in training) with 0
    X_predict = X_predict.reindex(columns=valid_cols, fill_value=0)

    print("Predicting scores...")
    predict_df["predicted_rating"] = rf.predict(X_predict)

    # --- Output ---

    # Filter: Only recommend games with > 50 ratings (Avoids unknown garbage)
    min_votes = 50
    recommendations = predict_df[
        (predict_df["ratings_count"] > min_votes)
    ].sort_values("predicted_rating", ascending=False).head(20)

    print("\n" + "=" * 40)
    print("      YOUR PERSONAL RECOMMENDATIONS      ")
    print("=" * 40)

    cols_to_show = ["name", "predicted_rating", "released_year", "genres"]
    print(recommendations[cols_to_show].to_string(index=False))

    # Save to CSV
    out_file = PROCESSED_DIR.parent.parent / "scripts" / "objective3_outputs" / "my_recommendations.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(out_file, index=False)
    print(f"\nSaved list to: {out_file}")


if __name__ == "__main__":
    main()
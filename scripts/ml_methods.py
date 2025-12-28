"""
Objective 3: Machine Learning Taste Model
- Task: Supervised Regression (Predict 'my_rating')
- Models: Ridge (Linear), Lasso (Feature Selection), Random Forest (Non-linear)
- Goal: Interpretability & Validation of previous findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import re
import ast

# --- Configuration ---
# Adjust paths if needed
RATINGS_FILE = "data/processed/my_ratings_template.csv"
RAWG_FILE = "data/processed/rawg_games_filtered_10k.csv"

OUT_DIR = Path("scripts/objective3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Blacklist (Redundant if using clean file, but safe to keep)
BLACKLIST_TAGS = {"sexual content", "nudity", "nsfw", "hentai", "mature", "dating sim"}
MIN_FEAT_FREQ = 5


# --- Utilities ---

def parse_multilabel(x):
    """Parses stringified lists or pipe-separated strings."""
    if pd.isna(x): return []
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
        if t_clean and t_clean.lower() not in BLACKLIST_TAGS:
            clean.append(t_clean)
    return sorted(list(set(clean)))


def normalize_name(s):
    return re.sub(r"[^a-z0-9\s]+", " ", str(s).lower()).strip()


def load_data():
    # Robust loading
    try:
        ratings = pd.read_csv(RATINGS_FILE)
        rawg = pd.read_csv(RAWG_FILE)
    except FileNotFoundError:
        # Fallback
        ratings = pd.read_csv(Path(RATINGS_FILE).name)
        rawg = pd.read_csv(Path(RAWG_FILE).name)

    # Merge
    if "id" in ratings.columns and "id" in rawg.columns:
        merged = ratings.merge(rawg, on="id", how="inner", suffixes=("_r", "_g"))
    else:
        ratings["_norm"] = ratings["name"].apply(normalize_name)
        rawg["_norm"] = rawg["name"].apply(normalize_name)
        merged = ratings.merge(rawg, on="_norm", how="inner", suffixes=("_r", "_g"))

    return merged


# --- Main Logic ---

def main():
    merged = load_data()
    print(f"Merged Data: {len(merged)} rows")

    # Target Selection
    target_col = next((c for c in ["my_rating_10", "my_rating", "rating", "score", "my_rating_10_r"]
                       if c in merged.columns), None)

    if not target_col:
        print("Error: No target column found.")
        return

    df = merged.dropna(subset=[target_col]).copy()
    df["target"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["target"])
    print(f"Modeling with {len(df)} rated games.")

    # --- Feature Engineering ---

    # 1. Genres & Tags (Categorical -> One-Hot)
    genres = df.get("genres", df.get("genres_g", "")).apply(parse_multilabel)
    tags = df.get("tags", df.get("tags_g", "")).apply(parse_multilabel)

    # Prefix features so we know what they are
    labels = (genres.apply(lambda x: [f"GENRE_{i}" for i in x]) +
              tags.apply(lambda x: [f"TAG_{i}" for i in x])).tolist()

    mlb = MultiLabelBinarizer()
    X_cat = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_, index=df.index)

    # Drop Rare Features (Noise Reduction)
    keep_cols = X_cat.columns[X_cat.sum() >= MIN_FEAT_FREQ]
    X_cat = X_cat[keep_cols]

    # 2. Numerical Features
    # Release Year
    y_col = next((c for c in ["released_year", "released_year_g", "year"] if c in df.columns), None)
    if y_col:
        # Simple Imputation
        years = pd.to_numeric(df[y_col], errors="coerce").fillna(2010)
        X_cat["NUM_Year"] = years

    # Metacritic (External Quality Signal)
    if "metacritic" in df.columns:
        meta = pd.to_numeric(df["metacritic"], errors="coerce").fillna(df["metacritic"].median())
        X_cat["NUM_Metacritic"] = meta

    # Ratings Count (Popularity) - Log Scaled
    rc_col = "ratings_count"
    if rc_col in df.columns:
        rc = pd.to_numeric(df[rc_col], errors="coerce").fillna(0)
        X_cat["NUM_LogRatingsCount"] = np.log1p(rc)

    X = X_cat
    y = df["target"]

    print(f"Features: {X.shape[1]}")

    # --- Train/Test Split ---
    # 75/25 is standard for small datasets to ensure the test set is representative
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scaling (Important for Ridge/Lasso)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # --- Modeling ---

    models = {
        "Ridge (L2)": Ridge(alpha=10.0),  # High alpha for stronger regularization
        "Lasso (L1)": Lasso(alpha=0.1),  # L1 penalty zeroes out coefficients
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    }

    results = []

    for name, model in models.items():
        # Train
        if "Forest" in name:
            model.fit(X_train, y_train)  # Trees work fine unscaled
            preds = model.predict(X_test)
            train_preds = model.predict(X_train)
        else:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            train_preds = model.predict(X_train_s)

        # Eval
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "Train MAE": mean_absolute_error(y_train, train_preds)
        })

        # --- Visualization (Ridge Only) ---
        if name == "Ridge (L2)":
            # 1. Actual vs Predicted
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, preds, alpha=0.6, color='purple')
            # Perfect prediction line
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            plt.xlabel("Actual Rating")
            plt.ylabel("Predicted Rating")
            plt.title(f"Ridge Regression (MAE={mae:.2f})")
            plt.savefig(OUT_DIR / "ridge_actual_vs_pred.png")
            plt.close()

            # 2. Top Coefficients
            coefs = pd.Series(model.coef_, index=X.columns).sort_values()
            # Top 10 Positive and Top 10 Negative
            top_coefs = pd.concat([coefs.head(10), coefs.tail(10)])

            plt.figure(figsize=(10, 8))
            colors = ['#ff6666' if x < 0 else '#66b3ff' for x in top_coefs.values]
            plt.barh(top_coefs.index, top_coefs.values, color=colors)
            plt.axvline(0, color='black', linewidth=0.8)
            plt.title("Model Coefficients: Strongest Predictors of Taste")
            plt.xlabel("Coefficient Value")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "ridge_coefficients.png")
            plt.close()

    # --- Save Metrics ---
    res_df = pd.DataFrame(results)
    print("\n--- Model Results ---")
    print(res_df)
    res_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    print(f"\nOutputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
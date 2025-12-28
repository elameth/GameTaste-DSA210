"""
Objective 2: Exploratory Data Analysis & Hypothesis Testing
- Analyzes rating distribution (Skewness, Normality).
- Performs Mann-Whitney U tests on Genres & Tags to find significant preferences.
- Analyzes Release Year trends (Era buckets).
- Computes correlations (Metacritic vs Personal Rating).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, skew
from pathlib import Path
import re
import ast

# --- Configuration ---
# Update these paths if your files are in a different folder
RATINGS_FILE = "data/processed/my_ratings_template.csv"
RAWG_FILE = "data/processed/rawg_games_filtered_10k.csv"

OUT_DIR = Path("scripts/objective2_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)



# --- Utilities ---

def parse_multilabel(x):
    """Robustly parses stringified lists or pipe-separated strings."""
    if pd.isna(x): return []

    # Try parsing as python list string "['Action', 'RPG']"
    if str(x).startswith("["):
        try:
            res = ast.literal_eval(x)
        except:
            res = []
    else:
        # Fallback to pipe or comma split
        res = re.split(r"[|,;]", str(x))

    clean = []
    for t in res:
        t_clean = str(t).strip()
        # Remove empty strings
        if t_clean and t_clean.lower():
            clean.append(t_clean)
    return sorted(list(set(clean)))


def normalize_name(s):
    """Normalizes game names for merging."""
    return re.sub(r"[^a-z0-9\s]+", " ", str(s).lower()).strip()


def load_and_merge():
    print(f"Loading {RATINGS_FILE} and {RAWG_FILE}...")
    try:
        ratings = pd.read_csv(RATINGS_FILE)
        rawg = pd.read_csv(RAWG_FILE)
    except FileNotFoundError:
        # Fallback for when running in the same dir
        ratings = pd.read_csv(Path(RATINGS_FILE).name)
        rawg = pd.read_csv(Path(RAWG_FILE).name)

    # 1. Merge by ID if available (Best)
    if "id" in ratings.columns and "id" in rawg.columns:
        merged = ratings.merge(rawg, on="id", how="inner", suffixes=("_r", "_g"))
    else:
        # 2. Merge by Name (Fallback)
        ratings["_norm"] = ratings["name"].apply(normalize_name)
        rawg["_norm"] = rawg["name"].apply(normalize_name)
        merged = ratings.merge(rawg, on="_norm", how="inner", suffixes=("_r", "_g"))

    return merged


# --- Main Analysis ---

def main():
    merged = load_and_merge()
    print(f"Merged {len(merged)} games.")

    # Identify the Rating Column
    # It might be 'my_rating_10', 'rating', 'score', or suffixed with _r
    candidates = ["my_rating_10", "my_rating", "rating", "score", "my_rating_10_r"]
    target_col = next((c for c in candidates if c in merged.columns), None)

    if not target_col:
        print("Error: Could not find a rating column (e.g., my_rating_10).")
        return

    # Filter valid ratings
    df = merged.dropna(subset=[target_col]).copy()
    df["my_rating"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["my_rating"])
    print(f"Analyzing {len(df)} rated games.")

    # --- 1. Distribution Analysis ---
    mean_rating = df["my_rating"].mean()
    rating_skew = skew(df["my_rating"])

    print("\n--- 1. Rating Distribution ---")
    print(f"Mean Rating: {mean_rating:.2f}")
    print(f"Skewness: {rating_skew:.2f} (Close to 0 is normal, negative is generous)")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["my_rating"], kde=True, bins=10, discrete=True, color='skyblue')
    plt.axvline(mean_rating, color='r', linestyle='--', label=f'Mean ({mean_rating:.1f})')
    plt.title(f"Distribution of My Ratings (Skew={rating_skew:.2f})")
    plt.xlabel("Rating (1-10)")
    plt.legend()
    plt.savefig(OUT_DIR / "rating_distribution.png")
    plt.close()

    # --- 2. Hypothesis Testing (Mann-Whitney U) ---
    print("\n--- 2. Hypothesis Testing (Tags/Genres) ---")

    # Flatten all tags/genres to count them
    # We create a 'long' dataframe where each row is (GameID, Feature, Rating)
    expanded_rows = []
    for idx, row in df.iterrows():
        g_raw = row.get("genres", row.get("genres_g", ""))
        t_raw = row.get("tags", row.get("tags_g", ""))

        # Combine genres and tags
        features = parse_multilabel(g_raw) + parse_multilabel(t_raw)

        for f in features:
            expanded_rows.append({"idx": idx, "feature": f, "rating": row["my_rating"]})

    feat_df = pd.DataFrame(expanded_rows)

    # Only test features that appear in at least 5 games (Statistical Relevance)
    counts = feat_df["feature"].value_counts()
    valid_feats = counts[counts >= 5].index
    feat_df = feat_df[feat_df["feature"].isin(valid_feats)]

    results = []
    for feat in valid_feats:
        # Get ratings for games WITH this feature
        # vs games WITHOUT this feature
        has_feat_indices = feat_df[feat_df["feature"] == feat]["idx"].unique()

        group_yes = df.loc[df.index.isin(has_feat_indices), "my_rating"]
        group_no = df.loc[~df.index.isin(has_feat_indices), "my_rating"]

        if len(group_yes) < 3 or len(group_no) < 3: continue

        # Mann-Whitney U Test
        stat, p = mannwhitneyu(group_yes, group_no, alternative='two-sided')
        diff = group_yes.mean() - group_no.mean()

        results.append({
            "Feature": feat,
            "Count": len(group_yes),
            "Diff": diff,  # Positive means you like it more than average
            "p_value": p
        })

    res_df = pd.DataFrame(results).sort_values("p_value")

    # Save all results
    res_df.to_csv(OUT_DIR / "hypothesis_tests.csv", index=False)

    # Filter for significant ones (p < 0.10 for EDA purposes)
    sig_df = res_df[res_df["p_value"] < 0.10].copy()
    print(f"Found {len(sig_df)} statistically significant features.")
    print(sig_df[["Feature", "Diff", "p_value", "Count"]].head(10))

    # Plot Top Significant Features
    if not sig_df.empty:
        # Take top 10 pos and top 10 neg by Diff magnitude, or just top significant
        top_sig = sig_df.sort_values("Diff", key=abs, ascending=False).head(20).sort_values("Diff")

        plt.figure(figsize=(10, 8))
        colors = ['#ff6666' if x < 0 else '#66b3ff' for x in top_sig['Diff']]
        plt.barh(top_sig["Feature"], top_sig["Diff"], color=colors)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title("Statistically Significant Preferences (p < 0.10)")
        plt.xlabel("Rating Difference (Avg with feature - Avg without)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "significant_features.png")
        plt.close()

    # --- 3. Era Analysis ---
    print("\n--- 3. Era Analysis ---")
    y_col = "released_year" if "released_year" in df.columns else (
        "released_year_g" if "released_year_g" in df.columns else "year")

    if y_col in df.columns:
        df["year"] = pd.to_numeric(df[y_col], errors="coerce")
        # Filter mostly valid years (e.g. 1980+)
        valid_years = df[(df["year"] > 1980) & (df["year"] < 2026)].copy()

        # Bucket into Eras
        bins = [1990, 2000, 2010, 2015, 2020, 2026]
        labels = ["90s", "00s", "Early 10s", "Late 10s", "20s"]
        valid_years["Era"] = pd.cut(valid_years["year"], bins=bins, labels=labels)

        # Stats
        era_stats = valid_years.groupby("Era")["my_rating"].agg(['count', 'mean', 'std'])
        print(era_stats)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Era", y="my_rating", data=valid_years, palette="Set2")
        plt.title("Rating Distribution by Era")
        plt.ylabel("My Rating")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUT_DIR / "rating_by_era.png")
        plt.close()

    # --- 4. Correlations ---
    print("\n--- 4. Correlation Matrix ---")
    # Identify numeric columns of interest
    candidates = ["my_rating", "metacritic", "rating", "year"]
    # 'rating' in RAWG is the community score (out of 5 usually)

    corr_cols = [c for c in candidates if c in df.columns]

    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr()
        print(corr_matrix)

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "correlation_matrix.png")
        plt.close()

    print(f"\nDone! All plots saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
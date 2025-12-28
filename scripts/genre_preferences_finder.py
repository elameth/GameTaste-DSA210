"""
Objective 1 -:
- Properly splits multi-genre strings.
- Keeps GENRE and TAG separated.
- Tall, readable graphs.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

try:
    from scipy.stats import mannwhitneyu
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------------------------
# Configuration
# ---------------------------



# [FIX] Increased threshold to remove "noisy" tags
# Now a tag must appear in at least 10 games to be considered a "Trend".
MIN_TAG_FREQ = 7
MIN_GENRE_FREQ = 2
MIN_FEAT_FREQ = 5


# ---------------------------
# Paths
# ---------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
if not PROCESSED_DIR.exists():
    PROCESSED_DIR = Path.cwd()

RATINGS_PATH = PROCESSED_DIR / "my_ratings_template.csv"
RAWG_PATH = PROCESSED_DIR / "rawg_games_filtered_10k.csv"

OUT_DIR = PROJECT_ROOT / "scripts" / "objective1_outputs"
if not OUT_DIR.parent.exists():
    OUT_DIR = Path.cwd() / "objective1_outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------

def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_multilabel(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    if isinstance(x, (list, set, tuple)):
        tokens = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return []

        tokens = None
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    tokens = [str(t).strip() for t in v if str(t).strip()]
            except Exception:
                pass
            if tokens is None:
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, list):
                        tokens = [str(t).strip() for t in v if str(t).strip()]
                except Exception:
                    pass

        if tokens is None:
            parts = re.split(r"[|,;]", s)
            tokens = [p.strip() for p in parts if p.strip()]

    # Normalize tokens
    out = []
    for t in tokens:
        t_clean = t.replace("/", " ").replace("-", " ")
        t_clean = re.sub(r"\s+", " ", t_clean).strip()

        if t_clean and t_clean.lower():
            out.append(t_clean)

    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0:
        return np.nan
    return (ma - mb) / pooled


def extract_year(series: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().mean() >= 0.8:
        yr = s_num.round().astype("Int64")
        yr = yr.where((yr >= 1970) & (yr <= 2035))
        return yr

    dt = pd.to_datetime(series, errors="coerce")
    yr = dt.dt.year.astype("Int64")
    yr = yr.where((yr >= 1970) & (yr <= 2035))
    return yr


def col_in_merged(base: Optional[str], merged: pd.DataFrame) -> Optional[str]:
    if base is None:
        return None
    if base in merged.columns:
        return base
    if f"{base}_r" in merged.columns:
        return f"{base}_r"
    if f"{base}_g" in merged.columns:
        return f"{base}_g"
    return None


# ---------------------------
# Column detection
# ---------------------------

@dataclass
class Cols:
    rating: str
    rating_id: Optional[str]
    rating_name: Optional[str]
    rawg_id: Optional[str]
    rawg_name: Optional[str]
    rawg_genres: str
    rawg_tags: Optional[str]
    rawg_year: Optional[str]


def detect_columns(ratings: pd.DataFrame, rawg: pd.DataFrame) -> Cols:
    rating_col = find_col(ratings, [
        "my_rating_10", "my_rating", "rating", "score", "personal_rating"
    ])
    if rating_col is None:
        bad = {"id", "game_id", "rawg_id", "status"}
        best = None
        for c in ratings.columns:
            if c.lower() in bad:
                continue
            s = pd.to_numeric(ratings[c], errors="coerce")
            frac = s.notna().mean()
            if frac >= 0.8 and (best is None or frac > best[1]):
                best = (c, frac)
        if best is None:
            raise ValueError("Could not detect rating column.")
        rating_col = best[0]
        print(f"[Auto-detect] Using numeric ratings column: {rating_col}")

    return Cols(
        rating=rating_col,
        rating_id=find_col(ratings, ["id", "rawg_id", "game_id"]),
        rating_name=find_col(ratings, ["name", "title", "game_name"]),
        rawg_id=find_col(rawg, ["id", "rawg_id", "game_id"]),
        rawg_name=find_col(rawg, ["name", "title", "game_name"]),
        rawg_genres=find_col(rawg, ["genres", "genre"]),
        rawg_tags=find_col(rawg, ["tags", "tag"]),
        rawg_year=find_col(rawg, ["released_year", "released", "year"]),
    )


# ---------------------------
# Pipeline
# ---------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Cols]:
    # Robust load
    r_path = RATINGS_PATH if RATINGS_PATH.exists() else Path(RATINGS_PATH.name)
    g_path = RAWG_PATH if RAWG_PATH.exists() else Path(RAWG_PATH.name)

    assert r_path.exists(), f"Missing: {r_path}"
    assert g_path.exists(), f"Missing: {g_path}"

    ratings = pd.read_csv(r_path)
    rawg = pd.read_csv(g_path)
    cols = detect_columns(ratings, rawg)
    return ratings, rawg, cols


def merge_data(ratings: pd.DataFrame, rawg: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    merged = None
    miss_rate = 1.0

    if cols.rating_id and cols.rawg_id:
        r = ratings.copy()
        g = rawg.copy()
        r[cols.rating_id] = pd.to_numeric(r[cols.rating_id], errors="coerce")
        g[cols.rawg_id] = pd.to_numeric(g[cols.rawg_id], errors="coerce")
        merged = r.merge(g, left_on=cols.rating_id, right_on=cols.rawg_id,
                         how="left", suffixes=("_r", "_g"))
        check = cols.rawg_name or cols.rawg_id
        check_col = col_in_merged(check, merged)
        if check_col is not None:
            miss_rate = merged[check_col].isna().mean()

    if merged is None or miss_rate > 0.50:
        r = ratings.copy()
        g = rawg.copy()
        r["_norm_name"] = r[cols.rating_name].astype(str).map(normalize_name)
        g["_norm_name"] = g[cols.rawg_name].astype(str).map(normalize_name)
        merged = r.merge(g, on="_norm_name", how="left", suffixes=("_r", "_g"))

    return merged


def build_Xy(merged: pd.DataFrame, cols: Cols) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rating_col = col_in_merged(cols.rating, merged)
    genres_col = col_in_merged(cols.rawg_genres, merged)
    tags_col = col_in_merged(cols.rawg_tags, merged)
    year_col = col_in_merged(cols.rawg_year, merged)

    df = merged.dropna(subset=[rating_col, genres_col]).copy()
    y = pd.to_numeric(df[rating_col], errors="coerce")
    df = df[y.notna()].copy()
    y = y.loc[df.index].astype(float)

    genres = df[genres_col].map(parse_multilabel)
    tags = df[tags_col].map(parse_multilabel) if tags_col else pd.Series([[]] * len(df), index=df.index)

    # Filter with updated thresholds
    all_genres = pd.Series([t for row in genres for t in row])
    keep_genres = set(all_genres.value_counts()[all_genres.value_counts() >= MIN_GENRE_FREQ].index)
    genres = genres.map(lambda row: [t for t in row if t in keep_genres])

    all_tags = pd.Series([t for row in tags for t in row])
    keep_tags = set(all_tags.value_counts()[all_tags.value_counts() >= MIN_TAG_FREQ].index)
    tags = tags.map(lambda row: [t for t in row if t in keep_tags])

    labels = (genres.map(lambda r: [f"GENRE: {t}" for t in r]) +
              tags.map(lambda r: [f"TAG: {t}" for t in r])).tolist()

    mlb = MultiLabelBinarizer(sparse_output=False)
    X_cat = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_, index=df.index)

    # Rare drop
    freq = X_cat.sum(axis=0)
    keep = freq[freq >= MIN_FEAT_FREQ].index
    X_cat = X_cat.loc[:, keep]

    # Year
    year_f = pd.Series([0.0]*len(df), index=df.index)
    if year_col:
        year = extract_year(df[year_col])
        year_f = year.astype("float").fillna(year.median())

    X = X_cat.copy()
    X["NUM: release_year"] = year_f

    return X, y, df


def compute_stats(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    global_mean = float(y.mean())
    y_np = y.to_numpy(dtype=float)
    rows = []

    for feat in X.columns:
        if feat.startswith("NUM: "):
            x = X[feat].to_numpy(dtype=float)
            if np.std(x) == 0: continue
            corr = float(np.corrcoef(x, y_np)[0, 1])
            rows.append({
                "feature": feat, "type": "numeric", "group": "NUM",
                "count_with": len(x), "mean_with": np.nan, "lift": np.nan,
                "corr_with_rating": corr
            })
            continue

        mask = X[feat].to_numpy(dtype=int) == 1
        with_feat = y_np[mask]
        without_feat = y_np[~mask]

        if with_feat.size < MIN_FEAT_FREQ: continue

        mean_with = float(with_feat.mean())
        lift = mean_with - global_mean
        d = cohens_d(with_feat, without_feat)

        group = "GENRE" if "GENRE:" in feat else "TAG"

        rows.append({
            "feature": feat, "type": "binary", "group": group,
            "count_with": int(mask.sum()), "mean_with": mean_with,
            "lift": lift, "cohens_d": d, "corr_with_rating": np.nan
        })

    out = pd.DataFrame(rows)
    # Sort
    out_bin = out[out["type"] == "binary"].sort_values("lift", ascending=False)
    out_num = out[out["type"] == "numeric"].sort_values("corr_with_rating", ascending=False)
    return pd.concat([out_bin, out_num], axis=0).reset_index(drop=True)


def plot_top_lifts(stats: pd.DataFrame, out_path: Path, top_n: int = 20, group: Optional[str] = None) -> None:
    s = stats[(stats["type"] == "binary")].copy()
    if group is not None:
        s = s[s["group"] == group]
    if s.empty: return

    # Top Positive and Top Negative
    pos = s[s["lift"] > 0].sort_values("lift", ascending=False).head(top_n)
    neg = s[s["lift"] < 0].sort_values("lift", ascending=True).head(top_n)

    # Sort strictly ascending (Negative -> Positive)
    p = pd.concat([pos, neg], axis=0).sort_values("lift", ascending=True)

    plt.figure(figsize=(10, max(8, len(p) * 0.35)))
    colors = ['#ff6666' if x < 0 else '#66b3ff' for x in p["lift"]]
    plt.barh(p["feature"], p["lift"], color=colors)
    plt.axvline(0.0, color='black', linewidth=0.8)

    title = f"Top Rating Lift (min {MIN_TAG_FREQ} games)"
    if group: title += f" - {group}"
    plt.title(title)
    plt.xlabel("Lift")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_year_relationship(X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    col = "NUM: release_year"
    if col not in X.columns:
        return

    x_vals = X[col].to_numpy()
    y_vals = y.to_numpy()

    # Calculate Correlation
    valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    if valid_mask.sum() < 2:
        corr = 0.0
    else:
        corr = np.corrcoef(x_vals[valid_mask], y_vals[valid_mask])[0, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, alpha=0.5, label='Data points')

    # Add Trendline
    if valid_mask.sum() >= 2:
        z = np.polyfit(x_vals[valid_mask], y_vals[valid_mask], 1)
        p = np.poly1d(z)
        # Sort for clean line plotting
        sort_idx = np.argsort(x_vals[valid_mask])
        plt.plot(x_vals[valid_mask][sort_idx], p(x_vals[valid_mask][sort_idx]),
                 "r--", linewidth=2, label=f'Trend (r={corr:.2f})')

    plt.title(f"My Rating vs Release Year (Correlation: {corr:.2f})")
    plt.xlabel("Release year")
    plt.ylabel("My rating")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main() -> None:
    ratings, rawg, cols = load_data()
    merged = merge_data(ratings, rawg, cols)

    print(f"Merged rows: {len(merged)}")

    X, y, df_used = build_Xy(merged, cols)

    print(f"Used rows: {len(y)}")
    print(f"Features: {X.shape[1]}")

    stats = compute_stats(X, y)
    stats.to_csv(OUT_DIR / "feature_stats.csv", index=False)

    print("Generating plots...")
    plot_top_lifts(stats, OUT_DIR / "top_lifts_overall.png", top_n=20)
    plot_top_lifts(stats, OUT_DIR / "top_lifts_genres.png", top_n=15, group="GENRE")
    plot_top_lifts(stats, OUT_DIR / "top_lifts_tags.png", top_n=15, group="TAG")
    plot_year_relationship(X, y, OUT_DIR / "rating_vs_year.png")

    print("\n=== Top Positive GENRES ===")
    print(stats[stats["group"] == "GENRE"].head(10)[["feature", "count_with", "lift"]])

    print("\n=== Top Positive TAGS ===")
    print(stats[stats["group"] == "TAG"].head(10)[["feature", "count_with", "lift"]])

    print("\nOutputs saved to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
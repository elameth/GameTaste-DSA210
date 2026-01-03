# scripts/export_my_ratings_template.py
#
# Reads your RAWG export CSV (elamethy.csv), drops only "Not played" rows
# , resolves each game via RAWG API,
# and writes a clean rating template: data/my_ratings_template.csv
#
# Output columns: id, name, status, my_rating_10
#
# Usage (from project root):
#   set RAWG_API_KEY=...   (CMD)  OR  $env:RAWG_API_KEY="..." (PowerShell)
#   python scripts/export_my_ratings_template.py
#
import os
import re
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAWG_KEY = os.environ.get("RAWG_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

IN_PATH = DATA_DIR / "elamethy.csv"
OUT_PATH = DATA_DIR / "my_ratings_template.csv"

RAWG_API_BASE = "https://api.rawg.io/api"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; GameTasteDSA210/1.0)",
        "Accept": "application/json",
    })
    retry = Retry(
        total=6,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def slug_from_url(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    m = re.search(r"rawg\.io/games/([^/?#]+)", url)
    return m.group(1) if m else None


def normalize_status(x) -> str:
    # Keep empty/NaN as empty string (you said those are often played due to a bug)
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def should_exclude(status: str) -> bool:
    # Your CSV statuses (confirmed): Completed, Played, Currently playing, Uncategorized, Not played, empty
    # Rule: remove only "Not played", keep empty.
    return status.strip().lower() == "not played"


def get_game_by_slug(session: requests.Session, slug: str) -> Optional[Dict[str, Any]]:
    url = f"{RAWG_API_BASE}/games/{slug}"
    r = session.get(url, params={"key": RAWG_KEY}, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None


def search_game(session: requests.Session, name: str) -> Optional[Dict[str, Any]]:
    # Fallback: search by name and take the top result
    url = f"{RAWG_API_BASE}/games"
    r = session.get(url, params={"key": RAWG_KEY, "search": name, "page_size": 1}, timeout=30)
    if r.status_code != 200:
        return None
    results = (r.json() or {}).get("results") or []
    return results[0] if results else None


def main():
    if not RAWG_KEY:
        raise RuntimeError("RAWG_API_KEY environment variable is not set.")

    if not IN_PATH.exists():
        raise RuntimeError(f"Input file not found: {IN_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    if "Game" not in df.columns or "Url" not in df.columns or "Status" not in df.columns:
        raise RuntimeError(f"Expected columns Game, Url, Status in {IN_PATH}, got: {list(df.columns)}")

    # Filter: remove only "Not played"; keep empty Status
    df["Status_norm"] = df["Status"].apply(normalize_status)
    df = df[~df["Status_norm"].apply(should_exclude)].copy()

    s = make_session()

    rows: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()

    # tiny cache so we don't hit API multiple times for identical slugs/names
    slug_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    name_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    for _, r in df.iterrows():
        name = str(r.get("Game", "")).strip()
        status = str(r.get("Status_norm", "")).strip()
        url = str(r.get("Url", "")).strip()
        slug = slug_from_url(url) if url else None

        game = None

        if slug:
            if slug not in slug_cache:
                slug_cache[slug] = get_game_by_slug(s, slug)
                time.sleep(0.02)  # tiny politeness delay; safe to set 0.0
            game = slug_cache[slug]

        if game is None and name:
            if name not in name_cache:
                name_cache[name] = search_game(s, name)
                time.sleep(0.02)
            game = name_cache[name]

        if game is None:
            print(f"[WARN] Could not resolve: {name} | status='{status}' | url='{url}'")
            continue

        gid = game.get("id")
        gname = game.get("name") or name
        if not isinstance(gid, int):
            print(f"[WARN] Bad id for: {name} -> {gid}")
            continue

        if gid in seen_ids:
            continue
        seen_ids.add(gid)

        rows.append({
            "id": gid,
            "name": gname,
            "status": status,
            "my_rating_10": "",  # you fill this later (1–10)
        })

    # Write output
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "status", "my_rating_10"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["name"].lower()))

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

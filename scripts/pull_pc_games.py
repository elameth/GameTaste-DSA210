import os
import time
import json
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
from dateutil import tz

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_BASE_URL = "https://api.igdb.com/v4"

PC_PLATFORM_ID = 6  # Microsoft Windows (PC)

RAW_OUT = "data/raw/igdb_pc_games_r500.jsonl"
CSV_OUT = "data/processed/igdb_pc_games_r500.csv"


def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def get_app_access_token(client_id: str, client_secret: str, timeout: int = 30) -> str:
    r = requests.post(
        TWITCH_TOKEN_URL,
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def igdb_post(
    endpoint: str,
    client_id: str,
    token: str,
    query: str,
    timeout: int = 30,
    retries: int = 6,
) -> List[Dict[str, Any]]:
    url = f"{IGDB_BASE_URL}/{endpoint}"
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    for attempt in range(retries):
        resp = requests.post(url, headers=headers, data=query.encode("utf-8"), timeout=timeout)

        # Token expired/invalid -> refresh handled outside if needed
        if resp.status_code == 401:
            raise RuntimeError("401 Unauthorized from IGDB (token invalid/expired).")

        # Rate limited -> backoff
        if resp.status_code == 429 and attempt < retries - 1:
            sleep_s = 1.0 + attempt * 1.5
            print(f"[rate-limit] 429 received; sleeping {sleep_s:.1f}s and retrying...")
            time.sleep(sleep_s)
            continue

        # Other errors
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"IGDB request failed after {retries} attempts.")


def unix_to_year(u: Optional[int]) -> Optional[int]:
    if not u:
        return None
    try:
        return time.gmtime(int(u)).tm_year
    except Exception:
        return None


def unix_to_iso(u: Optional[int]) -> Optional[str]:
    if not u:
        return None
    try:
        # Store as UTC ISO string for reproducibility
        return time.strftime("%Y-%m-%d", time.gmtime(int(u)))
    except Exception:
        return None


def flatten_game(g: Dict[str, Any]) -> Dict[str, Any]:
    def names(obj_list: Any) -> str:
        if not isinstance(obj_list, list):
            return ""
        out = []
        for x in obj_list:
            if isinstance(x, dict) and "name" in x and x["name"]:
                out.append(str(x["name"]))
        return "|".join(sorted(set(out)))

    def ids(obj_list: Any) -> str:
        if not isinstance(obj_list, list):
            return ""
        out = []
        for x in obj_list:
            if isinstance(x, dict) and "id" in x and x["id"] is not None:
                out.append(str(x["id"]))
        return "|".join(sorted(set(out), key=lambda s: int(s)))

    # involved_companies.company.name
    companies = []
    inv = g.get("involved_companies")
    if isinstance(inv, list):
        for item in inv:
            if isinstance(item, dict):
                c = item.get("company")
                if isinstance(c, dict) and c.get("name"):
                    companies.append(str(c["name"]))
    companies_s = "|".join(sorted(set(companies)))

    # cover url
    cover_url = None
    cover = g.get("cover")
    if isinstance(cover, dict):
        cover_url = cover.get("url")

    row = {
        "igdb_id": g.get("id"),
        "name": g.get("name"),
        "first_release_date_unix": g.get("first_release_date"),
        "first_release_date": unix_to_iso(g.get("first_release_date")),
        "release_year": unix_to_year(g.get("first_release_date")),
        # ratings
        "user_rating": g.get("rating"),
        "user_rating_count": g.get("rating_count"),
        "critic_rating": g.get("aggregated_rating"),
        "critic_rating_count": g.get("aggregated_rating_count"),
        # tags/features
        "genres": names(g.get("genres")),
        "themes": names(g.get("themes")),
        "keywords": names(g.get("keywords")),
        "game_modes": names(g.get("game_modes")),
        "player_perspectives": names(g.get("player_perspectives")),
        "platforms": names(g.get("platforms")),
        "platform_ids": ids(g.get("platforms")),
        "companies": companies_s,
        "cover_url": cover_url,
        "url": g.get("url"),
    }
    return row


def main():
    client_id = get_env("TWITCH_CLIENT_ID")
    client_secret = get_env("TWITCH_CLIENT_SECRET")

    token = get_app_access_token(client_id, client_secret)
    print("[ok] got app access token")

    # You can increase LIMIT to 500; IGDB commonly supports 500
    LIMIT = 500
    offset = 0

    #we’ll pull “main games” only (category=0), PC platform, and >=500 user votes
    # Also request the tag fields we’ll use for ML.
    fields = (
        "id,name,url,first_release_date,"
        "rating,rating_count,aggregated_rating,aggregated_rating_count,"
        "genres.name,themes.name,keywords.name,game_modes.name,player_perspectives.name,"
        "platforms.id,platforms.name,"
        "involved_companies.company.name,"
        "cover.url"
    )

    total_rows = 0
    flat_rows: List[Dict[str, Any]] = []

    # Overwrite old outputs if they exist
    if os.path.exists(RAW_OUT):
        os.remove(RAW_OUT)

    while True:
        query = f"""
        fields {fields};
        where platforms = {PC_PLATFORM_ID} & rating_count >= 500;
        sort rating_count desc;
        limit {LIMIT};
        offset {offset};
        """

        batch = igdb_post("games", client_id, token, query)

        if not batch:
            print("[done] no more results")
            break

        # Append raw JSONL
        with open(RAW_OUT, "a", encoding="utf-8") as f:
            for g in batch:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

        # Flatten for CSV
        for g in batch:
            flat_rows.append(flatten_game(g))

        total_rows += len(batch)
        print(f"[pull] offset={offset}  batch={len(batch)}  total={total_rows}")

        # next page
        offset += LIMIT

        # friendly pacing (IGDB rate limit is strict)
        time.sleep(0.35)  # ~3 req/sec max


    df = pd.DataFrame(flat_rows)

    # Some light cleanup
    df = df.drop_duplicates(subset=["igdb_id"]).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    print(f"[saved] raw: {RAW_OUT}")
    print(f"[saved] csv: {CSV_OUT}")
    if df.empty:
        print("[warn] dataframe is empty (0 rows). Check filters.")
    else:
        print(df.head(5)[["igdb_id", "name", "user_rating", "user_rating_count", "critic_rating"]])


if __name__ == "__main__":
    main()

import os
import csv
import time
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAWG_KEY = os.getenv("RAWG_API_KEY")
BASE_URL = "https://api.rawg.io/api"

OUT_CSV = Path("data/rawg_games_filtered_10k.csv")
PROGRESS_FILE = Path("data/rawg_progress.txt")
SEEN_IDS_FILE = Path("data/rawg_seen_ids.txt")

FIELDNAMES = [
    "id",
    "name",
    "released",
    "released_year",
    "metacritic",
    "rating",          # RAWG aggregate numeric rating (derived from categorical votes)
    "ratings_count",
    "genres",
    "tags",
]

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

def passes_filter(g: dict) -> bool:
    return g.get("metacritic") is not None


def safe_join(items, key="name", sep="|") -> str:
    if not items:
        return ""
    out = []
    for x in items:
        v = x.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return sep.join(out)

def load_int_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    s = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s.add(int(line))
            except ValueError:
                pass
    return s

def append_seen_id(path: Path, game_id: int):
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{game_id}\n")

def read_progress() -> int:
    if not PROGRESS_FILE.exists():
        return 1
    try:
        return int(PROGRESS_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return 1

def write_progress(page: int):
    PROGRESS_FILE.write_text(str(page), encoding="utf-8")

def ensure_csv_has_header():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        return
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()

def count_rows_in_csv() -> int:
    if not OUT_CSV.exists():
        return 0
    with OUT_CSV.open("r", encoding="utf-8", newline="") as f:
        # subtract header
        return max(sum(1 for _ in f) - 1, 0)

def fetch_page(session: requests.Session, page: int, page_size: int, ordering: str) -> list[dict]:
    url = f"{BASE_URL}/games"
    params = {
        "key": RAWG_KEY,
        "page": page,
        "page_size": page_size,
        "ordering": ordering,
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} on {r.url}\n{r.text[:400]}")
    data = r.json()
    return data.get("results") or []

def main(
    max_games: int = 10_000,
    ordering: str = "-added",
    page_size: int = 40,
    sleep_s: float = 0.0,
):
    if not RAWG_KEY:
        raise RuntimeError("RAWG_API_KEY env var not set")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_has_header()

    seen_ids = load_int_set(SEEN_IDS_FILE)  # optional but helpful
    start_page = read_progress()

    already = count_rows_in_csv()
    print(f"Already have {already} rows in {OUT_CSV}")
    print(f"Resuming from page {start_page} (seen_ids={len(seen_ids)})")

    s = make_session()
    written = 0
    page = start_page

    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)

        while (already + written) < max_games:
            results = fetch_page(s, page=page, page_size=page_size, ordering=ordering)

            if not results:
                print("No more results returned by API; stopping.")
                break

            # Save progress *as soon as the page is successfully fetched*
            write_progress(page + 1)

            for g in results:
                gid = g.get("id")
                if gid is None:
                    continue

                # De-dup protection (rare but cheap)
                try:
                    gid_int = int(gid)
                except Exception:
                    continue
                if gid_int in seen_ids:
                    continue

                if not passes_filter(g):
                    continue

                released = g.get("released")
                released_year = int(released[:4]) if released else None

                row = {
                    "id": gid_int,
                    "name": g.get("name"),
                    "released": released,
                    "released_year": released_year,
                    "metacritic": g.get("metacritic"),
                    "rating": g.get("rating"),
                    "ratings_count": g.get("ratings_count"),
                    "genres": safe_join(g.get("genres") or [], key="name"),
                    "tags": safe_join(g.get("tags") or [], key="name"),
                }

                w.writerow(row)
                written += 1
                seen_ids.add(gid_int)
                append_seen_id(SEEN_IDS_FILE, gid_int)

                if (already + written) >= max_games:
                    break

            if (already + written) % 200 == 0:
                print(f"Progress: {already + written}/{max_games} rows (last page={page})")

            page += 1
            time.sleep(sleep_s)

    print(f"Done. Added {written} new rows. Total now ≈ {already + written}.")
    print(f"CSV: {OUT_CSV}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Seen IDs file: {SEEN_IDS_FILE}")

if __name__ == "__main__":
    main()

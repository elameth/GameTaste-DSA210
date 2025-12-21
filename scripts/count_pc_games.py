import os
import time
import requests

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_GAMES_URL = "https://api.igdb.com/v4/games"

CLIENT_ID = os.environ["TWITCH_CLIENT_ID"]
CLIENT_SECRET = os.environ["TWITCH_CLIENT_SECRET"]

def get_token():
    r = requests.post(
        TWITCH_TOKEN_URL,
        params={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def count_where(where: str, limit: int = 500) -> int:
    token = get_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    total = 0
    offset = 0
    while True:
        q = f"""
fields id;
where {where};
limit {limit};
offset {offset};
"""
        r = requests.post(IGDB_GAMES_URL, headers=headers, data=q, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        total += len(batch)
        offset += limit
        if offset % (limit * 10) == 0:
            print(f"[count] offset={offset} total={total}")
        time.sleep(0.35)
    return total

def main():
    base = "aggregated_rating != null & aggregated_rating_count >= 10"
    non_mobile = base + " & platforms != 34 & platforms != 39"
    non_mobile_main = non_mobile + " & category = 0"

    print("Critic-rated (>=10) ALL platforms:", count_where(base))
    print("Critic-rated (>=10) excluding mobile:", count_where(non_mobile))
    print("Critic-rated (>=10) excluding mobile + main games:", count_where(non_mobile_main))

if __name__ == "__main__":
    main()

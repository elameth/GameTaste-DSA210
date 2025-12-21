import os
import requests

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_GAMES_URL = "https://api.igdb.com/v4/games"

CLIENT_ID = os.environ["TWITCH_CLIENT_ID"]
CLIENT_SECRET = os.environ["TWITCH_CLIENT_SECRET"]

def get_token() -> str:
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

def main():
    token = get_token()

    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    # Apicalypse query body (this is how IGDB works)
    body = """
    fields id, name, rating, rating_count, first_release_date;
    where rating != null & rating_count >= 500;
    sort rating desc;
    limit 10;
    """

    r = requests.post(IGDB_GAMES_URL, headers=headers, data=body, timeout=30)
    r.raise_for_status()
    print(r.json())

if __name__ == "__main__":
    main()

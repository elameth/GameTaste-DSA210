import os, requests

TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
CLIENT_ID = os.environ["TWITCH_CLIENT_ID"]
CLIENT_SECRET = os.environ["TWITCH_CLIENT_SECRET"]

def token():
    r = requests.post(
        TWITCH_TOKEN_URL,
        params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "grant_type": "client_credentials"},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

t = token()
headers = {"Client-ID": CLIENT_ID, "Authorization": f"Bearer {t}", "Accept": "application/json"}

q = """
fields id,name,platforms.name;
where platforms = 6;
limit 10;
"""
r = requests.post("https://api.igdb.com/v4/games", headers=headers, data=q, timeout=30)
r.raise_for_status()
print(r.json())

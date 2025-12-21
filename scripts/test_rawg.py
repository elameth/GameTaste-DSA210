import os, requests
key = os.environ["RAWG_API_KEY"]
r = requests.get("https://api.rawg.io/api/games",
                 params={"key": key, "metacritic": "1,100", "page_size": 1},
                 timeout=30)
r.raise_for_status()
print("metacritic games count =", r.json()["count"])

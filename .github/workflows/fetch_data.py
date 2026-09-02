import requests
import pandas as pd
import os
from datetime import datetime, timedelta

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

def fetch_competition_matches(competition_code, days_past=30, days_future=0):
    headers = {"X-Auth-Token": API_KEY}
    date_from = (datetime.now() - timedelta(days=days_past)).strftime("%Y-%m-%d")
    date_to = (datetime.now() + timedelta(days=days_future)).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")
    matches = response.json()["matches"]
    rows = []
    for m in matches:
        if m["score"]["winner"] is None:
            continue
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        home_goals = m["score"]["fullTime"]["home"]
        away_goals = m["score"]["fullTime"]["away"]
        rows.append({"home_team": home, "away_team": away,
                     "home_goals": home_goals, "away_goals": away_goals,
                     "date": m["utcDate"][:10]})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = fetch_competition_matches("PL", days_past=100, days_future=0)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/matches.csv", index=False)
    print(f"Fetched {len(df)} matches")

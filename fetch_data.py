import requests
import pandas as pd
import os
from datetime import datetime, timedelta

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

# 你要抓取的联赛代码列表（可自行增删）
LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]  # 英超、西甲、德甲、意甲、法甲

def fetch_league_matches(league_code, days_past=100):
    headers = {"X-Auth-Token": API_KEY}
    date_from = (datetime.now() - timedelta(days=days_past)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "status": "FINISHED",
        "limit": 100
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"联赛 {league_code} 请求失败，状态码 {response.status_code}")
        return pd.DataFrame()
    
    matches = response.json()["matches"]
    rows = []
    for m in matches:
        if m["score"]["winner"] is None:
            continue
        rows.append({
            "league": league_code,
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "date": m["utcDate"][:10]
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    all_frames = []
    for league in LEAGUES:
        print(f"正在抓取 {league} ...")
        df = fetch_league_matches(league, days_past=100)
        if not df.empty:
            all_frames.append(df)
    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        os.makedirs("data", exist_ok=True)
        final_df.to_csv("data/matches.csv", index=False)
        print(f"共抓取 {len(final_df)} 场比赛")
    else:
        print("未抓取到任何比赛数据")

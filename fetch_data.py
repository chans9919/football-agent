import requests
import pandas as pd
import os
from datetime import datetime

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

def fetch_odds():
    """从 The Odds API 获取未来比赛的赔率（胜平负）"""
    if not ODDS_API_KEY:
        print("未设置 ODDS_API_KEY，跳过赔率抓取")
        return pd.DataFrame()
    sports = "soccer_epl,soccer_spain_la_liga,soccer_germany_bundesliga,soccer_italy_serie_a,soccer_france_ligue_one"
    url = f"https://api.the-odds-api.com/v4/sports/{sports}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"赔率请求失败，状态码 {response.status_code}")
        return pd.DataFrame()
    data = response.json()
    rows = []
    for match in data:
        home = match["home_team"]
        away = match["away_team"]
        commence_time = match["commence_time"]
        best_home = best_draw = best_away = None
        for bk in match.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    if best_home is None or outcomes.get(home, 999) < best_home:
                        best_home = outcomes.get(home)
                        best_draw = outcomes.get("Draw")
                        best_away = outcomes.get(away)
        if best_home is not None:
            rows.append({
                "home_team": home,
                "away_team": away,
                "commence_time": commence_time,
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("历史比赛数据由人工维护，本脚本仅更新赔率数据。")
    odds_df = fetch_odds()
    if not odds_df.empty:
        os.makedirs("data", exist_ok=True)
        odds_df.to_csv("data/odds.csv", index=False)
        print(f"赔率数据已更新：{len(odds_df)} 条")
    else:
        print("未抓取到赔率数据（可能未设置 ODDS_API_KEY 或额度耗尽）")

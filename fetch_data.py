import requests
import pandas as pd
import os
from datetime import datetime, timedelta

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")  # 如果没设置就是空字符串
BASE_URL = "https://api.football-data.org/v4"

LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]

def fetch_league_matches(league_code, days_past=800):
    """
    抓取指定联赛过去 days_past 天的已结束比赛。
    注意：免费版 API 单次请求最多返回 100 场比赛，可能无法覆盖 800 天全部数据。
    如果需要更多历史，建议手动上传数据。
    """
    headers = {"X-Auth-Token": API_KEY}
    date_from = (datetime.now() - timedelta(days=days_past)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "status": "FINISHED",
        "limit": 100          # 免费版最大 100
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
    all_frames = []
    for league in LEAGUES:
        print(f"正在抓取 {league} 历史比赛（过去 800 天）...")
        df = fetch_league_matches(league, days_past=800)
        if not df.empty:
            all_frames.append(df)
        else:
            print(f"联赛 {league} 没有抓取到数据")
    
    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        os.makedirs("data", exist_ok=True)
        final_df.to_csv("data/matches.csv", index=False)
        print(f"共抓取 {len(final_df)} 场历史比赛，已保存到 data/matches.csv")
    else:
        print("未抓取到任何历史比赛数据，保留原有 data/matches.csv（如果有）")

    # 抓取赔率
    odds_df = fetch_odds()
    if not odds_df.empty:
        os.makedirs("data", exist_ok=True)
        odds_df.to_csv("data/odds.csv", index=False)
        print(f"抓取到 {len(odds_df)} 场比赛的赔率，已保存到 data/odds.csv")
    else:
        print("未抓取到赔率数据（可能未设置 ODDS_API_KEY 或 API 额度耗尽）")

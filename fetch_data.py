import requests
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from team_config import normalize_team_name

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
BASE_FOOTBALL_URL = "https://api.football-data.org/v4"
BASE_ODDS_URL = "https://api.the-odds-api.com/v4/sports"

# 联赛映射：football-data编码 → 你的标准编码
LEAGUE_MAP = {
    "PL": "PL",
    "PD": "PD",
    "BL1": "BL1",
    "SA": "SA",
    "FL1": "FL1"
}

# 联赛映射：你的标准编码 → The Odds API 运动标识
LEAGUE_ODDS_SPORT = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_1"
}

def fetch_finished_matches(days=7):
    """拉取最近已结束的比赛，更新历史库"""
    if not FOOTBALL_DATA_KEY:
        print("未设置FOOTBALL_DATA_API_KEY，跳过赛果更新")
        return
    
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    all_matches = []
    
    for league_code in LEAGUE_MAP.keys():
        try:
            date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
            
            url = f"{BASE_FOOTBALL_URL}/competitions/{league_code}/matches"
            params = {
                "dateFrom": date_from,
                "dateTo": date_to,
                "status": "FINISHED",
                "limit": 100
            }
            
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            
            for m in data["matches"]:
                home = normalize_team_name(m["homeTeam"]["name"])
                away = normalize_team_name(m["awayTeam"]["name"])
                hg = m["score"]["fullTime"]["home"]
                ag = m["score"]["fullTime"]["away"]
                
                if hg is not None and ag is not None:
                    all_matches.append({
                        "date": m["utcDate"][:10],
                        "league": LEAGUE_MAP[league_code],
                        "home_team": home,
                        "away_team": away,
                        "home_goals": int(hg),
                        "away_goals": int(ag)
                    })
                    
            print(f"✅ {league_code} 拉取到 {len([m for m in all_matches if m['league']==LEAGUE_MAP[league_code]])} 场已赛")
            
        except Exception as e:
            print(f"⚠️ {league_code} 赛果拉取失败: {str(e)}")
            continue
    
    if not all_matches:
        return
    
    new_df = pd.DataFrame(all_matches)
    data_path = "data/matches.csv"
    
    if os.path.exists(data_path):
        old_df = pd.read_csv(data_path)
        if "league" not in old_df.columns:
            old_df["league"] = "PL"
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "league", "home_team", "away_team"], 
            keep="last"
        )
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("date").reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    combined.to_csv(data_path, index=False, encoding="utf-8")
    print(f"📊 历史库更新完成，总计 {len(combined)} 场比赛")

def fetch_odds():
    """从 The Odds API 获取未来比赛的赔率（胜平负），分联赛单独请求"""
    if not ODDS_API_KEY:
        print("未设置 ODDS_API_KEY，跳过赔率抓取")
        return pd.DataFrame()
    
    all_odds = []
    
    for league, sport in LEAGUE_ODDS_SPORT.items():
        try:
            url = f"{BASE_ODDS_URL}/{sport}/odds"
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "uk,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            }
            
            response = requests.get(url, params=params, timeout=20)
            if response.status_code != 200:
                print(f"⚠️ {league} 赔率请求失败，状态码 {response.status_code}")
                continue
            
            data = response.json()
            
            for match in data:
                home = normalize_team_name(match["home_team"])
                away = normalize_team_name(match["away_team"])
                commence_time = match["commence_time"]
                
                # 收集所有庄家赔率，取中位数过滤异常值
                all_home = []
                all_draw = []
                all_away = []
                
                for bk in match.get("bookmakers", []):
                    for market in bk.get("markets", []):
                        if market["key"] == "h2h":
                            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                            home_odds = outcomes.get(match["home_team"])
                            draw_odds = outcomes.get("Draw")
                            away_odds = outcomes.get(match["away_team"])
                            
                            if home_odds and draw_odds and away_odds:
                                all_home.append(home_odds)
                                all_draw.append(draw_odds)
                                all_away.append(away_odds)
                
                if all_home:
                    best_home = float(np.median(all_home))
                    best_draw = float(np.median(all_draw))
                    best_away = float(np.median(all_away))
                    
                    all_odds.append({
                        "league": league,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence_time,
                        "odds_home": best_home,
                        "odds_draw": best_draw,
                        "odds_away": best_away
                    })
            
            print(f"✅ {league} 拉取到 {len([o for o in all_odds if o['league']==league])} 场赔率")
            
        except Exception as e:
            print(f"⚠️ {league} 赔率拉取异常: {str(e)}")
            continue
    
    return pd.DataFrame(all_odds)

if __name__ == "__main__":
    print("===== 1. 更新比赛结果 =====")
    fetch_finished_matches(days=7)
    
    print("\n===== 2. 更新赔率数据 =====")
    odds_df = fetch_odds()
    if not odds_df.empty:
        os.makedirs("data", exist_ok=True)
        odds_df.to_csv("data/odds.csv", index=False, encoding="utf-8")
        print(f"✅ 赔率数据已更新：{len(odds_df)} 条")
    else:
        print("⚠️ 未抓取到赔率数据（可能未设置 ODDS_API_KEY 或额度耗尽）")

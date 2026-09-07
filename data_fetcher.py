import pandas as pd
from typing import List, Dict
from datetime import datetime
from team_name_normalizer import normalize_team_name

def load_matches_csv(csv_path: str = "data/matches.csv") -> List[Dict]:
    """加载比赛历史数据，清洗并标准化队名"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except:
        df = pd.read_csv(csv_path, encoding="gbk")
    
    matches = []
    for _, row in df.iterrows():
        home_team = normalize_team_name(str(row.get("HomeTeam", "")))
        away_team = normalize_team_name(str(row.get("AwayTeam", "")))
        
        # 半场数据有效性检查
        has_half_time = pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))
        
        match = {
            "date": datetime.strptime(str(row["Date"]), "%Y-%m-%d").date(),
            "league": str(row.get("League", "default")),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0,
            "away_goals": int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0,
            "half_home_goals": int(row["HTHG"]) if has_half_time else 0,
            "half_away_goals": int(row["HTAG"]) if has_half_time else 0,
            "has_half_time": has_half_time
        }
        matches.append(match)
    
    return sorted(matches, key=lambda x: x["date"])

def load_odds_csv(csv_path: str = "data/odds.csv") -> Dict:
    """加载赔率数据，取最新记录"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except:
        return {}
    
    odds_map = {}
    # 按时间倒序，保留最新一条
    df = df.sort_values("fetched_at", ascending=False)
    for _, row in df.iterrows():
        key = (str(row["date"]), str(row["home_team"]), str(row["away_team"]))
        if key not in odds_map:
            odds_map[key] = {
                "home_odds": float(row.get("home_odds", 0)),
                "draw_odds": float(row.get("draw_odds", 0)),
                "away_odds": float(row.get("away_odds", 0)),
                "over_odds": float(row.get("over_odds", 0)),
                "under_odds": float(row.get("under_odds", 0))
            }
    return odds_map

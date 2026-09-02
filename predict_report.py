import pandas as pd
import numpy as np
import requests
import os
from train_model import train_poisson, predict_match_prob

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

def get_upcoming_matches(competition_code, days_ahead=1):
    headers = {"X-Auth-Token": API_KEY}
    date_from = pd.Timestamp.now().strftime("%Y-%m-%d")
    date_to = (pd.Timestamp.now() + pd.Timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return []
    matches = response.json()["matches"]
    upcoming = []
    for m in matches:
        upcoming.append({"home_team": m["homeTeam"]["name"],
                         "away_team": m["awayTeam"]["name"],
                         "date": m["utcDate"]})
    return upcoming

def generate_report():
    df = pd.read_csv("data/matches.csv")
    upcoming = get_upcoming_matches("PL", days_ahead=2)
    if not upcoming:
        return "No upcoming matches"
    
    report = "# 足球预测报告\n\n"
    for match in upcoming:
        home = match["home_team"]
        away = match["away_team"]
        try:
            lh, la = train_poisson(df, home, away)
        except:
            lh, la = 1.5, 1.2
        ph, pd_, pa = predict_match_prob(lh, la)
        report += f"## {home} vs {away}\n"
        report += f"时间: {match['date']}\n\n"
        report += f"- 主胜概率: {ph:.1%}\n"
        report += f"- 平局概率: {pd_:.1%}\n"
        report += f"- 客胜概率: {pa:.1%}\n\n"
        # 简单推荐
        if ph > pd_ and ph > pa:
            rec = "主胜"
        elif pd_ > ph and pd_ > pa:
            rec = "平局"
        else:
            rec = "客胜"
        report += f"**推荐方向**: {rec}\n\n---\n\n"
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Report generated")

if __name__ == "__main__":
    generate_report()

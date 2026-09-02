import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from train_model import train_poisson, predict_match_prob

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

# 与 fetch_data.py 保持一致
LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]

# 联赛代码到中文名称的映射（可选，用于报告显示）
LEAGUE_NAMES = {
    "PL": "英超",
    "PD": "西甲",
    "BL1": "德甲",
    "SA": "意甲",
    "FL1": "法甲"
}

def get_upcoming_matches(league_code, days_ahead=3):
    headers = {"X-Auth-Token": API_KEY}
    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "status": "SCHEDULED",
        "limit": 100
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"联赛 {league_code} 未来赛事请求失败，状态码 {response.status_code}")
        return []
    matches = response.json()["matches"]
    upcoming = []
    for m in matches:
        upcoming.append({
            "league": league_code,
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "date": m["utcDate"]
        })
    return upcoming

def generate_report():
    if not os.path.exists("data/matches.csv"):
        return "没有历史数据文件 data/matches.csv"
    df = pd.read_csv("data/matches.csv")
    
    all_upcoming = []
    for league in LEAGUES:
        print(f"获取 {league} 未来赛程...")
        matches = get_upcoming_matches(league, days_ahead=3)
        all_upcoming.extend(matches)
    
    if not all_upcoming:
        return "未来 3 天内没有找到任何联赛的未开始比赛"
    
    report = "# 足球预测报告（多联赛）\n\n"
    report += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n\n"
    report += f"共 {len(all_upcoming)} 场比赛\n\n---\n\n"
    
    # 按联赛分组输出
    for league in LEAGUES:
        league_matches = [m for m in all_upcoming if m["league"] == league]
        if not league_matches:
            continue
        league_name = LEAGUE_NAMES.get(league, league)
        report += f"## {league_name}（{league}）\n\n"
        for match in league_matches:
            home = match["home_team"]
            away = match["away_team"]
            try:
                match_time = pd.to_datetime(match["date"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
            except:
                match_time = match["date"]
            
            # 从历史数据中筛选该联赛的比赛用于模型训练
            league_df = df[df["league"] == league]
            if league_df.empty:
                league_df = df  # 如果该联赛无历史数据，用全部数据兜底
            
            try:
                lh, la = train_poisson(league_df, home, away)
            except Exception as e:
                print(f"计算 {home} vs {away} 出错：{e}")
                lh, la = 1.5, 1.2
            
            ph, pd_, pa = predict_match_prob(lh, la)
            
            report += f"### {home} vs {away}\n\n"
            report += f"- 比赛时间：{match_time}\n"
            report += f"- 主胜概率：{ph:.1%}\n"
            report += f"- 平局概率：{pd_:.1%}\n"
            report += f"- 客胜概率：{pa:.1%}\n\n"
            if ph >= pd_ and ph >= pa:
                rec = "主胜"
                conf = ph
            elif pd_ >= ph and pd_ >= pa:
                rec = "平局"
                conf = pd_
            else:
                rec = "客胜"
                conf = pa
            report += f"**推荐方向**：{rec}（置信度 {conf:.1%}）\n\n---\n\n"
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已生成，共 {len(all_upcoming)} 场比赛")

if __name__ == "__main__":
    generate_report()

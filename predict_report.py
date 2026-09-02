import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from train_model import train_poisson, predict_match_prob

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]

LEAGUE_NAMES = {
    "PL": "英超",
    "PD": "西甲",
    "BL1": "德甲",
    "SA": "意甲",
    "FL1": "法甲"
}

# 常见球队中英文对照表（可自行补充）
TEAM_NAMES_ZH = {
    # 英超
    "Manchester City FC": "曼城",
    "Arsenal FC": "阿森纳",
    "Liverpool FC": "利物浦",
    "Chelsea FC": "切尔西",
    "Manchester United FC": "曼联",
    "Tottenham Hotspur FC": "热刺",
    "Newcastle United FC": "纽卡斯尔",
    "Brighton & Hove Albion FC": "布莱顿",
    "Aston Villa FC": "阿斯顿维拉",
    "West Ham United FC": "西汉姆",
    "Everton FC": "埃弗顿",
    "Wolverhampton Wanderers FC": "狼队",
    "Crystal Palace FC": "水晶宫",
    "Fulham FC": "富勒姆",
    "Brentford FC": "布伦特福德",
    "Leeds United FC": "利兹联",
    "Leicester City FC": "莱斯特城",
    "Southampton FC": "南安普顿",
    "Nottingham Forest FC": "诺丁汉森林",
    "AFC Bournemouth": "伯恩茅斯",
    # 西甲
    "Real Madrid CF": "皇家马德里",
    "FC Barcelona": "巴塞罗那",
    "Atlético Madrid": "马德里竞技",
    "Sevilla FC": "塞维利亚",
    "Real Sociedad": "皇家社会",
    "Real Betis": "皇家贝蒂斯",
    "Villarreal CF": "比利亚雷亚尔",
    "Valencia CF": "瓦伦西亚",
    "Athletic Club": "毕尔巴鄂竞技",
    "CA Osasuna": "奥萨苏纳",
    "Rayo Vallecano": "巴列卡诺",
    "RCD Espanyol": "西班牙人",
    "Getafe CF": "赫塔菲",
    "Cádiz CF": "加的斯",
    "UD Almería": "阿尔梅里亚",
    "Granada CF": "格拉纳达",
    "RC Celta de Vigo": "塞尔塔",
    "RCD Mallorca": "马洛卡",
    "Girona FC": "赫罗纳",
    # 德甲
    "FC Bayern München": "拜仁慕尼黑",
    "Borussia Dortmund": "多特蒙德",
    "RB Leipzig": "莱比锡红牛",
    "Bayer 04 Leverkusen": "勒沃库森",
    "Eintracht Frankfurt": "法兰克福",
    "VfL Wolfsburg": "沃尔夫斯堡",
    "Borussia Mönchengladbach": "门兴格拉德巴赫",
    "1. FSV Mainz 05": "美因茨",
    "SC Freiburg": "弗赖堡",
    "TSG 1899 Hoffenheim": "霍芬海姆",
    "1. FC Union Berlin": "柏林联合",
    "VfB Stuttgart": "斯图加特",
    "FC Augsburg": "奥格斯堡",
    "SV Werder Bremen": "云达不莱梅",
    "1. FC Köln": "科隆",
    "FC Schalke 04": "沙尔克04",
    "Hertha BSC": "柏林赫塔",
    "VfL Bochum": "波鸿",
    "1. FC Heidenheim": "海登海姆",
    "Darmstadt 98": "达姆施塔特",
    # 意甲
    "Juventus FC": "尤文图斯",
    "AC Milan": "AC米兰",
    "FC Internazionale Milano": "国际米兰",
    "SSC Napoli": "那不勒斯",
    "AS Roma": "罗马",
    "SS Lazio": "拉齐奥",
    "Atalanta BC": "亚特兰大",
    "ACF Fiorentina": "佛罗伦萨",
    "Torino FC": "都灵",
    "Bologna FC": "博洛尼亚",
    "Udinese Calcio": "乌迪内斯",
    "US Sassuolo Calcio": "萨索洛",
    "Hellas Verona FC": "维罗纳",
    "US Lecce": "莱切",
    "Cagliari Calcio": "卡利亚里",
    "Empoli FC": "恩波利",
    "Genoa CFC": "热那亚",
    "US Salernitana": "萨勒尼塔纳",
    "Frosinone Calcio": "弗罗西诺内",
    "AC Monza": "蒙扎",
    # 法甲
    "Paris Saint-Germain FC": "巴黎圣日耳曼",
    "Olympique de Marseille": "马赛",
    "Olympique Lyonnais": "里昂",
    "AS Monaco FC": "摩纳哥",
    "LOSC Lille": "里尔",
    "Stade Rennais FC": "雷恩",
    "OGC Nice": "尼斯",
    "RC Strasbourg Alsace": "斯特拉斯堡",
    "FC Nantes": "南特",
    "Montpellier HSC": "蒙彼利埃",
    "Stade Brestois 29": "布雷斯特",
    "Stade de Reims": "兰斯",
    "FC Lorient": "洛里昂",
    "Clermont Foot": "克莱蒙",
    "Toulouse FC": "图卢兹",
    "AJ Auxerre": "欧塞尔",
    "Angers SCO": "昂热",
    "FC Metz": "梅斯",
    "RC Lens": "朗斯",
    "Havre AC": "勒阿弗尔",
}

def get_team_name_zh(team_en):
    """将英文队名转为中文，未匹配则保留英文"""
    return TEAM_NAMES_ZH.get(team_en, team_en)

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
    
    for league in LEAGUES:
        league_matches = [m for m in all_upcoming if m["league"] == league]
        if not league_matches:
            continue
        league_name = LEAGUE_NAMES.get(league, league)
        report += f"## {league_name}（{league}）\n\n"
        for match in league_matches:
            home_en = match["home_team"]
            away_en = match["away_team"]
            home_zh = get_team_name_zh(home_en)
            away_zh = get_team_name_zh(away_en)
            
            try:
                match_time = pd.to_datetime(match["date"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
            except:
                match_time = match["date"]
            
            # 从历史数据中筛选该联赛的比赛用于模型训练
            league_df = df[df["league"] == league]
            if league_df.empty:
                league_df = df  # 如果该联赛无历史数据，用全部数据兜底
            
            try:
                lh, la = train_poisson(league_df, home_en, away_en)
            except Exception as e:
                print(f"计算 {home_en} vs {away_en} 出错：{e}")
                lh, la = 1.5, 1.2
            
            ph, pd_, pa = predict_match_prob(lh, la)
            
            report += f"### {home_zh} vs {away_zh}\n\n"
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

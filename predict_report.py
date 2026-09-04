import pandas as pd
import numpy as np
import requests
import os
import json
import traceback
from datetime import datetime, timedelta, timezone
from scipy.stats import poisson
from train_model import train_poisson, predict_match_prob, calculate_elo
from team_config import normalize_team_name

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

# ========== 竞彩时间窗口（北京时间 当日17:00 ~ 次日12:00） ==========
def get_time_window():
    now_utc = datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(timezone(timedelta(hours=8)))
    if now_bj.hour < 12:
        base_date = (now_bj - timedelta(days=1)).date()
    else:
        base_date = now_bj.date()
    start_bj = datetime.combine(
        base_date, 
        datetime.strptime("17:00", "%H:%M").time(),
        tzinfo=timezone(timedelta(hours=8))
    )
    end_bj = datetime.combine(
        base_date + timedelta(days=1), 
        datetime.strptime("12:00", "%H:%M").time(),
        tzinfo=timezone(timedelta(hours=8))
    )
    start_utc = start_bj.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_bj.astimezone(timezone.utc).replace(tzinfo=None)
    return start_utc, end_utc, base_date.strftime("%Y-%m-%d")

START_UTC, END_UTC, TARGET_DATE_LABEL = get_time_window()

LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]
LEAGUE_NAMES = {
    "PL": "英超",
    "PD": "西甲",
    "BL1": "德甲",
    "SA": "意甲",
    "FL1": "法甲"
}

# 分联赛双模型融合权重 [泊松, ELO]
LEAGUE_WEIGHTS = {
    "PL":  [0.35, 0.65],
    "PD":  [0.65, 0.35],
    "BL1": [0.35, 0.65],
    "SA":  [0.45, 0.55],
    "FL1": [0.50, 0.50],
}

# 分联赛三模型融合权重 [泊松, ELO, 市场]
LEAGUE_WEIGHTS_WITH_ODDS = {
    "PL":  [0.20, 0.50, 0.30],
    "PD":  [0.40, 0.25, 0.35],
    "BL1": [0.20, 0.50, 0.30],
    "SA":  [0.30, 0.35, 0.35],
    "FL1": [0.30, 0.30, 0.40],
}

# 中文名映射（部分，与原来相同，此处省略以免过长，你原有TEAM_NAMES_ZH保留不变）
TEAM_NAMES_ZH = {
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
    "Luton Town FC": "卢顿",
    "Burnley FC": "伯恩利",
    "Sheffield United FC": "谢菲联",
    "Ipswich Town FC": "伊普斯维奇",
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
    "Deportivo Alavés": "阿拉维斯",
    "UD Las Palmas": "拉斯帕尔马斯",
    "CD Leganés": "莱加内斯",
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
    "Holstein Kiel": "基尔",
    "FC St. Pauli": "圣保利",
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
    "Parma Calcio 1913": "帕尔马",
    "Como 1907": "科莫",
    "Venezia FC": "威尼斯",
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
    "AS Saint-Étienne": "圣埃蒂安",
    "Stade Malherbe Caen": "卡昂",
    "FC Girondins de Bordeaux": "波尔多",
    "ESTAC Troyes": "特鲁瓦",
    "Dijon FCO": "第戎",
}

def get_team_name_zh(team_en):
    if team_en in TEAM_NAMES_ZH:
        return TEAM_NAMES_ZH[team_en]
    team_lower = team_en.lower()
    if len(team_lower) >= 8:
        for en, zh in TEAM_NAMES_ZH.items():
            if en.lower() in team_lower or team_lower in en.lower():
                return zh
    return team_en

# ========== 以下函数与原代码相同（未改动） ==========
def get_upcoming_matches(league_code, days_ahead=2):
    if not API_KEY:
        print(f"⚠️ 未设置FOOTBALL_DATA_API_KEY，跳过{league_code}赛程拉取")
        return []
    headers = {"X-Auth-Token": API_KEY}
    date_from = datetime.utcnow().strftime("%Y-%m-%d")
    date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {
        "dateFrom": date_from,
        "dateTo": date_to,
        "status": "SCHEDULED",
        "limit": 100
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        matches = response.json()["matches"]
        upcoming = []
        for m in matches:
            home_en = normalize_team_name(m["homeTeam"]["name"])
            away_en = normalize_team_name(m["awayTeam"]["name"])
            upcoming.append({
                "league": league_code,
                "home_team": home_en,
                "away_team": away_en,
                "date": m["utcDate"]
            })
        print(f"✅ {league_code} 拉取到 {len(upcoming)} 场未来赛事")
        return upcoming
    except Exception as e:
        print(f"❌ {league_code} 未来赛事请求失败: {str(e)}")
        return []

def poisson_prob_matrix(lambda_home, lambda_away, max_goals=8):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    dc_draw_scores = [(0,0), (1,1), (2,2)]
    dc_other_scores = [(1,0), (0,1)]
    dc_draw_factor = 1.30
    dc_other_factor = 1.15
    for i, j in dc_draw_scores:
        if i < matrix.shape[0] and j < matrix.shape[1]:
            matrix[i, j] *= dc_draw_factor
    for i, j in dc_other_scores:
        if i < matrix.shape[0] and j < matrix.shape[1]:
            matrix[i, j] *= dc_other_factor
    matrix /= matrix.sum()
    return matrix

def match_probabilities(lambda_home, lambda_away):
    matrix = poisson_prob_matrix(lambda_home, lambda_away)
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    score_probs = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            score_probs[f"{i}-{j}"] = matrix[i,j]
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    lambda_half_home = lambda_home * 0.45
    lambda_half_away = lambda_away * 0.45
    half_matrix = poisson_prob_matrix(lambda_half_home, lambda_half_away, max_goals=4)
    ht_home = np.sum(np.tril(half_matrix, -1))
    ht_draw = np.sum(np.diag(half_matrix))
    ht_away = np.sum(np.triu(half_matrix, 1))
    htft_probs = {
        "胜胜": ht_home * home_win,
        "胜平": ht_home * draw,
        "胜负": ht_home * away_win,
        "平胜": ht_draw * home_win,
        "平平": ht_draw * draw,
        "平负": ht_draw * away_win,
        "负胜": ht_away * home_win,
        "负平": ht_away * draw,
        "负负": ht_away * away_win,
    }
    top_htft = sorted(htft_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    handicap_home_win = 0
    handicap_draw = 0
    handicap_away_win = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_i = i - 1
            if new_i < 0:
                handicap_away_win += matrix[i,j]
            else:
                if new_i > j:
                    handicap_home_win += matrix[i,j]
                elif new_i == j:
                    handicap_draw += matrix[i,j]
                else:
                    handicap_away_win += matrix[i,j]
    total_goals = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total = i + j
            total_goals[total] = total_goals.get(total, 0) + matrix[i,j]
    sorted_total = sorted(total_goals.items(), key=lambda x: x[0])
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "top_scores": top_scores,
        "top_htft": top_htft,
        "handicap": (handicap_home_win, handicap_draw, handicap_away_win),
        "total_goals": sorted_total
    }

def elo_probabilities(home_elo, away_elo, home_adv=65):
    diff = home_elo + home_adv - away_elo
    draw_prob = 0.30 - 0.07 * abs(diff) / 400
    draw_prob = max(0.20, min(0.32, draw_prob))
    expected_home_win = 1 / (1 + 10 ** (-diff / 400))
    remaining = 1 - draw_prob
    home_win = expected_home_win * remaining
    away_win = (1 - expected_home_win) * remaining
    total = home_win + draw_prob + away_win
    return home_win/total, draw_prob/total, away_win/total

def find_odds(odds_df, home_en, away_en):
    if odds_df.empty:
        return None
    match = odds_df[(odds_df["home_team"] == home_en) & (odds_df["away_team"] == away_en)]
    if not match.empty:
        return match.iloc[0]["odds_home"], match.iloc[0]["odds_draw"], match.iloc[0]["odds_away"]
    return None

def market_implied_prob(odds):
    raw = [1 / o for o in odds]
    total = sum(raw)
    return [p / total for p in raw]

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def fuse_probs(probs_list, weights):
    logits = [logit(p) for p in probs_list]
    fused_logit = sum(w * l for w, l in zip(weights, logits))
    return inv_logit(fused_logit)

def generate_match_id(date_str, home_team, away_team):
    def normalize(name):
        return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
    return f"{date_str}_{normalize(home_team)}_{normalize(away_team)}"

def load_or_calculate_elo(league_df, league):
    path = f"model/elo_{league}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            elo_dict = json.load(f)
        print(f"✅ {league} ELO模型加载成功，共{len(elo_dict)}支球队")
        return elo_dict
    print(f"⚠️ {league} 未找到预训练ELO文件，当场计算")
    if len(league_df) >= 10:
        elo_dict = calculate_elo(league_df)
        print(f"✅ {league} ELO当场计算完成，共{len(elo_dict)}支球队")
        return elo_dict
    else:
        print(f"⚠️ {league} 数据不足，ELO使用默认值")
        return {}

# ========== 新增：通俗解读函数 ==========
def generate_friendly_advice(home_zh, away_zh, lh, la,
                            p_poisson, p_elo, p_market,
                            poisson_probs, home_elo, away_elo,
                            fused_probs, odds_data, data_sufficient):
    advice = "\n━━━━ 通俗解读 ━━━━\n\n"
    
    # 比赛特征
    total_xg = lh + la
    if total_xg > 2.8:
        rhythm = "攻防节奏偏快，大球概率较高"
    elif total_xg < 2.3:
        rhythm = "攻防偏保守，小球概率较高"
    else:
        rhythm = "攻防节奏适中"
    elo_diff = home_elo - away_elo
    if abs(elo_diff) >= 150:
        level = "实力差距悬殊"
    elif abs(elo_diff) >= 80:
        level = "实力存在明显差距"
    elif abs(elo_diff) >= 30:
        level = "实力有一定差距"
    else:
        level = "实力非常接近"
    advice += f"🔍 比赛特征：{level}，总进球期望 {total_xg:.2f}，{rhythm}。\n"
    if not data_sufficient:
        advice += "   ⚠️ 联赛历史数据不足，实力评估仅供参考。\n"
    
    # 方向判断（红绿灯）
    max_prob = max(fused_probs)
    direction_idx = np.argmax(fused_probs)
    dir_map = {0: "主胜", 1: "平局", 2: "客胜"}
    direction_str = dir_map[direction_idx]
    if max_prob >= 0.5:
        light = "🟢 绿灯"
        grade = "A档（强推）"
    elif max_prob >= 0.4:
        light = "🟡 黄灯"
        grade = "B档（可买）"
    elif max_prob >= 0.35:
        light = "🟠 橙灯"
        grade = "C档（谨慎）"
    else:
        light = "🔴 红灯"
        grade = "不推荐"
    advice += f"📊 方向判断：{direction_str} {max_prob:.1%} {light} {grade}\n"
    # 模型一致性
    dirs = [np.argmax(p_poisson), np.argmax(p_elo)]
    if p_market:
        dirs.append(np.argmax(p_market))
    same_count = sum(1 for d in dirs if d == dirs[0])
    if same_count == len(dirs):
        advice += "   - 所有模型方向一致，分歧度低。\n"
    else:
        advice += "   - 模型存在一定分歧，需谨慎。\n"
    
    # 单场胜平负建议
    advice += "💰 单场胜平负建议："
    if grade in ["A档（强推）", "B档（可买）"]:
        advice += f"**{direction_str}**"
    elif grade == "C档（谨慎）":
        advice += f"**{direction_str}**（小注或观望）"
    else:
        advice += "无推荐"
    advice += "\n"
    if odds_data:
        odds_h, odds_d, odds_a = odds_data
        if direction_idx == 0:
            rec_odds = odds_h
        elif direction_idx == 1:
            rec_odds = odds_d
        else:
            rec_odds = odds_a
        # 赔率合理性
        if 1.5 <= rec_odds <= 2.2:
            odds_comment = "✅ 赔率合理"
        elif 1.3 <= rec_odds < 1.5:
            odds_comment = "⚠️ 赔率偏低，要求概率≥55%才买"
        elif 2.2 < rec_odds <= 3.0:
            odds_comment = "⚠️ 赔率偏高，要求概率≥50%才买"
        else:
            odds_comment = "❌ 赔率不合理，不建议"
        advice += f"   - 推荐赔率：{rec_odds:.2f}（{odds_comment}）\n"
        # 期望值简算
        ev = max_prob * rec_odds - 1
        advice += f"   - 预期：10场类似比赛约{max_prob*10:.0f}场命中，盈亏平衡点约{1/rec_odds:.1%}\n"
    else:
        advice += "   - 无赔率数据，无法判断赔率合理性。\n"
    
    # 串关适合度
    advice += "🔗 串关适合度："
    if grade in ["A档（强推）", "B档（可买）"] and odds_data:
        if 1.4 <= rec_odds <= 2.0:
            advice += "✅ 适合串关，可与另一场 1.5~1.8 的选项组合\n"
        else:
            advice += "⚠️ 赔率较高，串关需搭配更稳选项\n"
    else:
        advice += "❌ 不适合串关\n"
    
    # 让球建议
    hw, hd, ha = poisson_probs["handicap"]
    advice += "🎯 让球（主让一球）："
    if ha > 0.5:
        advice += f"客胜 {ha:.1%}，主队让球偏深，客队受让更稳\n"
    elif hw > 0.4:
        advice += f"主胜 {hw:.1%}，主队赢盘能力较强\n"
    else:
        advice += "三项接近，走盘风险高\n"
    
    # 半全场参考（强调半场平局概率大）
    top_htft = poisson_probs["top_htft"]
    advice += "⏱ 半全场参考：\n"
    advice += f"   最高概率组合：{top_htft[0][0]}（{top_htft[0][1]:.1%}），"
    if top_htft[0][0].startswith("平"):
        advice += "上半场平局概率大，比赛慢热。\n"
    else:
        advice += "上半场分胜负，节奏较快。\n"
    # 方向一致性建议
    dir_htft_map = {
        "home": ["胜胜", "平胜"],
        "draw": ["平平", "胜平", "负平"],
        "away": ["负负", "平负"]
    }
    consistent = dir_htft_map.get(["home","draw","away"][direction_idx], [])
    consistent_in_top = [c for c in consistent if any(t[0] == c for t in top_htft)]
    if consistent_in_top:
        advice += f"   与全场方向（{direction_str}）一致的组合：{', '.join(consistent_in_top)}，可优先关注。\n"
    else:
        advice += f"   与全场方向一致的组合未进入Top3，建议谨慎。\n"
    
    # 博冷提示
    if odds_data:
        # 找出概率最低但EV可能高的选项
        probs = fused_probs
        min_prob_idx = np.argmin(probs)
        min_prob = probs[min_prob_idx]
        if min_prob < 0.3:
            min_odds = [odds_data[0], odds_data[1], odds_data[2]][min_prob_idx]
            min_dir = ["主胜", "平局", "客胜"][min_prob_idx]
            advice += f"🎲 博冷提示：{min_dir} 概率仅 {min_prob:.1%}，赔率 {min_odds:.2f}，10场约中{min_prob*10:.0f}场，不建议主力投注。\n"
    
    advice += "\n"
    return advice

# ========== 主报告函数（修改部分） ==========
def generate_report():
    if not os.path.exists("data/matches.csv"):
        print("❌ 没有找到历史数据文件 data/matches.csv")
        return
    odds_df = pd.DataFrame()
    if os.path.exists("data/odds.csv"):
        odds_df = pd.read_csv("data/odds.csv")
        print(f"📊 加载赔率数据：{len(odds_df)} 条")
    else:
        print("⚠️ 未找到赔率文件")
    
    all_upcoming = []
    for league in LEAGUES:
        matches = get_upcoming_matches(league, days_ahead=2)
        all_upcoming.extend(matches)
    all_upcoming = [
        m for m in all_upcoming
        if START_UTC <= pd.to_datetime(m["date"], utc=True).tz_localize(None) <= END_UTC
    ]
    if not all_upcoming:
        print(f"❌ {TARGET_DATE_LABEL} 时段内无比赛可预测")
        return
    print(f"📅 {TARGET_DATE_LABEL} 时段内赛事：{len(all_upcoming)} 场")
    
    match_counter = 1
    predictions = []
    all_matches_df = pd.read_csv("data/matches.csv")
    print(f"📚 历史数据总场数：{len(all_matches_df)}")
    
    elo_cache = {}
    league_df_cache = {}
    data_sufficient_cache = {}
    for league in LEAGUES:
        if "league" in all_matches_df.columns:
            league_df = all_matches_df[all_matches_df["league"] == league].copy()
        else:
            league_df = all_matches_df.copy() if league == "PL" else pd.DataFrame()
        league_df_cache[league] = league_df
        elo_cache[league] = load_or_calculate_elo(league_df, league)
        data_sufficient_cache[league] = len(league_df) >= 10
    
    now_bj = datetime.utcnow() + timedelta(hours=8)
    report = "# 足球预测报告（多联赛）\n\n"
    report += f"生成时间：{now_bj.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n\n"
    report += f"预测时段：{TARGET_DATE_LABEL} 17:00 ~ 次日 12:00\n\n"
    report += f"数据来源：Football-Data.org + The Odds API + ELO\n\n"
    has_any_odds = any(find_odds(odds_df, m["home_team"], m["away_team"]) for m in all_upcoming)
    if has_any_odds:
        model_desc = "泊松模型（DC修正） + 全庄家赔率中位数 + ELO 在 logit 空间融合"
    else:
        model_desc = "泊松模型（DC修正） + ELO（本期无赔率数据）"
    report += f"模型说明：{model_desc}\n\n"
    report += f"共 {len(all_upcoming)} 场比赛\n\n---\n\n"
    
    all_upcoming_sorted = sorted(all_upcoming, key=lambda x: x["date"])
    current_league = ""
    for match in all_upcoming_sorted:
        league = match["league"]
        league_df = league_df_cache.get(league, pd.DataFrame())
        elo_dict = elo_cache.get(league, {})
        data_sufficient = data_sufficient_cache.get(league, False)
        league_name = LEAGUE_NAMES.get(league, league)
        
        if league != current_league:
            current_league = league
            if data_sufficient:
                report += f"\n## {league_name}\n\n"
            else:
                report += f"\n## {league_name} ⚠️ 数据不足，仅供参考\n\n"
        
        try:
            home_en = match["home_team"]
            away_en = match["away_team"]
            home_zh = get_team_name_zh(home_en)
            away_zh = get_team_name_zh(away_en)
            
            match_dt = pd.to_datetime(match["date"])
            if match_dt.tzinfo is None:
                match_dt = match_dt.tz_localize('UTC')
            match_time = match_dt.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
            
            lh, la = train_poisson(league_df, home_en, away_en)
            poisson_probs = match_probabilities(lh, la)
            p_poisson = [poisson_probs["home_win"], poisson_probs["draw"], poisson_probs["away_win"]]
            
            odds_data = find_odds(odds_df, home_en, away_en)
            if odds_data:
                p_market = market_implied_prob(odds_data)
            else:
                p_market = None
            
            home_elo = elo_dict.get(home_en, 1500)
            away_elo = elo_dict.get(away_en, 1500)
            p_elo = list(elo_probabilities(home_elo, away_elo))
            
            if p_market:
                weights = LEAGUE_WEIGHTS_WITH_ODDS.get(league, [0.3, 0.3, 0.4])
                probs_list = [p_poisson, p_elo, p_market]
            else:
                weights = LEAGUE_WEIGHTS.get(league, [0.5, 0.5])
                probs_list = [p_poisson, p_elo]
            total_w = sum(weights)
            weights_norm = [w / total_w for w in weights]
            fused_home = fuse_probs([p[0] for p in probs_list], weights_norm)
            fused_draw = fuse_probs([p[1] for p in probs_list], weights_norm)
            fused_away = fuse_probs([p[2] for p in probs_list], weights_norm)
            total_fused = fused_home + fused_draw + fused_away
            fused_home /= total_fused
            fused_draw /= total_fused
            fused_away /= total_fused
            fused_probs = [fused_home, fused_draw, fused_away]
            
            if fused_home >= fused_draw and fused_home >= fused_away:
                pred_direction = "home"
                conf = fused_home
            elif fused_draw >= fused_home and fused_draw >= fused_away:
                pred_direction = "draw"
                conf = fused_draw
            else:
                pred_direction = "away"
                conf = fused_away
            
            match_date_str = match_dt.strftime("%Y-%m-%d")
            match_id = generate_match_id(match_date_str, home_en, away_en)
            
            home_matches = len(league_df[league_df["home_team"] == home_en]) if len(league_df) > 0 else 0
            away_matches = len(league_df[league_df["away_team"] == away_en]) if len(league_df) > 0 else 0
            sample_count = min(home_matches, away_matches)
            
            record = {
                "match_id": match_id,
                "date": match_date_str,
                "league": league,
                "home_team": home_en,
                "away_team": away_en,
                "poisson_home": p_poisson[0],
                "poisson_draw": p_poisson[1],
                "poisson_away": p_poisson[2],
                "market_home": p_market[0] if p_market else np.nan,
                "market_draw": p_market[1] if p_market else np.nan,
                "market_away": p_market[2] if p_market else np.nan,
                "elo_home": p_elo[0],
                "elo_draw": p_elo[1],
                "elo_away": p_elo[2],
                "fused_home": fused_home,
                "fused_draw": fused_draw,
                "fused_away": fused_away,
                "pred_direction": pred_direction,
                "odds_home": odds_data[0] if odds_data else np.nan,
                "odds_draw": odds_data[1] if odds_data else np.nan,
                "odds_away": odds_data[2] if odds_data else np.nan,
                "ev_home": fused_home * odds_data[0] - 1 if odds_data else np.nan,
                "ev_draw": fused_draw * odds_data[1] - 1 if odds_data else np.nan,
                "ev_away": fused_away * odds_data[2] - 1 if odds_data else np.nan,
                "status": "pending"
            }
            predictions.append(record)
            
            match_no = f"{match_counter:03d}"
            match_counter += 1
            if data_sufficient:
                report += f"### {match_no} {home_zh} vs {away_zh}\n\n"
            else:
                report += f"### {match_no} {home_zh} vs {away_zh} ⚠️数据不足\n\n"
            report += f"- 开赛时间：{match_time}\n"
            report += f"- 期望进球：主 {lh:.2f}，客 {la:.2f}\n\n"
            report += f"**各模型概率**\n\n"
            report += f"| 模型 | 主胜 | 平局 | 客胜 |\n|---|---:|---:|---:|\n"
            report += f"| 泊松 | {p_poisson[0]:.1%} | {p_poisson[1]:.1%} | {p_poisson[2]:.1%} |\n"
            if p_market:
                report += f"| 市场 | {p_market[0]:.1%} | {p_market[1]:.1%} | {p_market[2]:.1%} |\n"
            else:
                report += f"| 市场 | 未匹配 | 未匹配 | 未匹配 |\n"
            report += f"| ELO  | {p_elo[0]:.1%} | {p_elo[1]:.1%} | {p_elo[2]:.1%} |\n\n"
            report += f"**融合后最终概率**\n"
            report += f"| 主胜 | 平局 | 客胜 |\n|---|---:|---:|\n"
            report += f"| {fused_home:.1%} | {fused_draw:.1%} | {fused_away:.1%} |\n\n"
            
            # 通俗解读（新增）
            report += generate_friendly_advice(
                home_zh, away_zh, lh, la,
                p_poisson, p_elo, p_market,
                poisson_probs, home_elo, away_elo,
                fused_probs, odds_data, data_sufficient
            )
            
            # 原有详细分析
            analysis_text = generate_match_analysis(
                home_zh, away_zh, lh, la,
                p_poisson, p_elo, p_market,
                poisson_probs, home_elo, away_elo,
                fused_probs, sample_count, data_sufficient
            )
            report += analysis_text
            report += "\n---\n\n"
        except Exception as e:
            print(f"❌ 跳过比赛 {home_en} vs {away_en}：{str(e)}")
            traceback.print_exc()
            continue
    
    # ===== 新增：今日二串一推荐 =====
    if len(predictions) >= 2:
        report += "\n## 🔗 今日二串一推荐\n\n"
        # 筛选可串场次：概率≥0.4 且赔率在1.4~2.2之间，或者概率≥0.5且赔率≤2.5
        candidates = []
        for p in predictions:
            max_prob = max(p["fused_home"], p["fused_draw"], p["fused_away"])
            direction = p["pred_direction"]
            odds = p.get(f"odds_{direction}", np.nan)
            if pd.isna(odds):
                continue
            if max_prob >= 0.4 and 1.4 <= odds <= 2.2:
                candidates.append(p)
            elif max_prob >= 0.5 and 1.3 <= odds <= 2.5:
                candidates.append(p)
        if len(candidates) >= 2:
            # 按赔率排序，简单组合两个
            candidates_sorted = sorted(candidates, key=lambda x: x[f"odds_{x['pred_direction']}"])
            # 生成两个组合
            combo1 = candidates_sorted[0:2]
            total_odds1 = combo1[0][f"odds_{combo1[0]['pred_direction']}"] * combo1[1][f"odds_{combo1[1]['pred_direction']}"]
            report += f"组合1（稳健型）：总赔率 {total_odds1:.2f}\n"
            for c in combo1:
                dir_zh = {"home": "主胜", "draw": "平局", "away": "客胜"}[c["pred_direction"]]
                report += f"  [{c['league']}] {get_team_name_zh(c['home_team'])} vs {get_team_name_zh(c['away_team'])} 推荐{dir_zh} 赔率{c[f'odds_{c[\"pred_direction\"]}']:.2f}\n"
            if len(candidates_sorted) >= 3:
                combo2 = [candidates_sorted[0], candidates_sorted[2]]
                total_odds2 = combo2[0][f"odds_{combo2[0]['pred_direction']}"] * combo2[1][f"odds_{combo2[1]['pred_direction']}"]
                report += f"组合2（进取型）：总赔率 {total_odds2:.2f}\n"
                for c in combo2:
                    dir_zh = {"home": "主胜", "draw": "平局", "away": "客胜"}[c["pred_direction"]]
                    report += f"  [{c['league']}] {get_team_name_zh(c['home_team'])} vs {get_team_name_zh(c['away_team'])} 推荐{dir_zh} 赔率{c[f'odds_{c[\"pred_direction\"]}']:.2f}\n"
            else:
                report += "（仅有两场可串，无第二组合）\n"
        else:
            report += "今天符合条件的可串比赛不足 2 场，建议休息。\n"
        report += "\n"
    
    print(f"\n✅ 成功生成 {match_counter-1} 场比赛预测")
    if predictions:
        new_pred_df = pd.DataFrame(predictions)
        os.makedirs("data", exist_ok=True)
        pred_path = "data/predictions.csv"
        if os.path.exists(pred_path):
            old_preds = pd.read_csv(pred_path)
            combined = pd.concat([old_preds, new_pred_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["match_id"], keep="last")
        else:
            combined = new_pred_df
        combined.to_csv(pred_path, index=False)
        print(f"💾 预测记录已保存，共 {len(combined)} 条")
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 报告已生成")

if __name__ == "__main__":
    generate_report()

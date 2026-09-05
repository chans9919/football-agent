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

def get_time_window():
    now_utc = datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(timezone(timedelta(hours=8)))
    if now_bj.hour < 12:
        base_date = (now_bj - timedelta(days=1)).date()
    else:
        base_date = now_bj.date()
    start_bj = datetime.combine(base_date, datetime.strptime("17:00", "%H:%M").time(), tzinfo=timezone(timedelta(hours=8)))
    end_bj = datetime.combine(base_date + timedelta(days=1), datetime.strptime("12:00", "%H:%M").time(), tzinfo=timezone(timedelta(hours=8)))
    start_utc = start_bj.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_bj.astimezone(timezone.utc).replace(tzinfo=None)
    return start_utc, end_utc, base_date.strftime("%Y-%m-%d")

START_UTC, END_UTC, TARGET_DATE_LABEL = get_time_window()

LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]
LEAGUE_NAMES = {"PL": "英超", "PD": "西甲", "BL1": "德甲", "SA": "意甲", "FL1": "法甲"}

LEAGUE_WEIGHTS = {
    "PL": [0.35, 0.65], "PD": [0.65, 0.35], "BL1": [0.35, 0.65], "SA": [0.45, 0.55], "FL1": [0.50, 0.50],
}
LEAGUE_WEIGHTS_WITH_ODDS = {
    "PL": [0.20, 0.50, 0.30], "PD": [0.40, 0.25, 0.35], "BL1": [0.20, 0.50, 0.30], "SA": [0.30, 0.35, 0.35], "FL1": [0.30, 0.30, 0.40],
}

LEAGUE_PARAMS = {
    "PL": {"draw_corr": 0.98, "tg_corr": 1.05, "single_draw_bonus": 1.08},
    "PD": {"draw_corr": 1.05, "tg_corr": 0.98, "single_draw_bonus": 1.08},
    "BL1": {"draw_corr": 0.95, "tg_corr": 1.08, "single_draw_bonus": 1.08},
    "SA": {"draw_corr": 1.03, "tg_corr": 0.97, "single_draw_bonus": 1.08},
    "FL1": {"draw_corr": 1.02, "tg_corr": 1.00, "single_draw_bonus": 1.08},
}

LEAGUE_TOTAL_ROUNDS = {"PL": 38, "PD": 38, "BL1": 34, "SA": 38, "FL1": 38}
LEAGUE_TEAM_COUNT = {"PL": 20, "PD": 20, "BL1": 18, "SA": 20, "FL1": 20}
LEAGUE_EUROPA_LINE = {"PL": 6, "PD": 6, "BL1": 6, "SA": 6, "FL1": 4}

MOTIVATION_SCORE = {"强": 15, "正常": 0, "弱": -10}
PING_STRICT_THRESHOLD = 0.48
PING_BALANCED_THRESHOLD = 0.44

TEAM_NAMES_ZH = {
    "Manchester City FC": "曼城", "Arsenal FC": "阿森纳", "Liverpool FC": "利物浦", "Chelsea FC": "切尔西",
    "Manchester United FC": "曼联", "Tottenham Hotspur FC": "热刺", "Newcastle United FC": "纽卡斯尔",
    "Brighton & Hove Albion FC": "布莱顿", "Aston Villa FC": "阿斯顿维拉", "West Ham United FC": "西汉姆",
    "Everton FC": "埃弗顿", "Wolverhampton Wanderers FC": "狼队", "Crystal Palace FC": "水晶宫",
    "Fulham FC": "富勒姆", "Brentford FC": "布伦特福德", "Leeds United FC": "利兹联",
    "Leicester City FC": "莱斯特城", "Southampton FC": "南安普顿", "Nottingham Forest FC": "诺丁汉森林",
    "AFC Bournemouth": "伯恩茅斯", "Ipswich Town FC": "伊普斯维奇", "Coventry City FC": "考文垂",
    "Sunderland AFC": "桑德兰", "Hull City AFC": "赫尔城",
    "Real Madrid CF": "皇家马德里", "FC Barcelona": "巴塞罗那", "Atlético Madrid": "马德里竞技",
    "Club Atlético de Madrid": "马德里竞技", "Sevilla FC": "塞维利亚", "Real Sociedad": "皇家社会",
    "Real Betis": "皇家贝蒂斯", "Real Betis Balompié": "皇家贝蒂斯", "Villarreal CF": "比利亚雷亚尔",
    "Valencia CF": "瓦伦西亚", "Athletic Club": "毕尔巴鄂竞技", "CA Osasuna": "奥萨苏纳",
    "Rayo Vallecano": "巴列卡诺", "Rayo Vallecano de Madrid": "巴列卡诺", "RCD Espanyol": "西班牙人",
    "RCD Espanyol de Barcelona": "西班牙人", "Getafe CF": "赫塔菲", "Cádiz CF": "加的斯",
    "UD Almería": "阿尔梅里亚", "Granada CF": "格拉纳达", "RC Celta de Vigo": "塞尔塔",
    "RCD Mallorca": "马洛卡", "Girona FC": "赫罗纳", "Deportivo Alavés": "阿拉维斯",
    "UD Las Palmas": "拉斯帕尔马斯", "CD Leganés": "莱加内斯", "Málaga CF": "马拉加",
    "Levante UD": "莱万特", "RC Deportivo La Coruña": "拉科鲁尼亚",
    "Real Racing Club de Santander": "桑坦德竞技",
    "FC Bayern München": "拜仁慕尼黑", "Borussia Dortmund": "多特蒙德", "RB Leipzig": "莱比锡红牛",
    "Bayer 04 Leverkusen": "勒沃库森", "Eintracht Frankfurt": "法兰克福", "VfL Wolfsburg": "沃尔夫斯堡",
    "Borussia Mönchengladbach": "门兴格拉德巴赫", "1. FSV Mainz 05": "美因茨", "SC Freiburg": "弗赖堡",
    "TSG 1899 Hoffenheim": "霍芬海姆", "1. FC Union Berlin": "柏林联合", "VfB Stuttgart": "斯图加特",
    "FC Augsburg": "奥格斯堡", "SV Werder Bremen": "云达不莱梅", "1. FC Köln": "科隆",
    "FC Schalke 04": "沙尔克04", "Hertha BSC": "柏林赫塔", "VfL Bochum": "波鸿",
    "1. FC Heidenheim": "海登海姆", "Darmstadt 98": "达姆施塔特", "Holstein Kiel": "基尔",
    "FC St. Pauli": "圣保利", "SV 07 Elversberg": "埃尔沃斯堡", "SC Paderborn 07": "帕德博恩",
    "Hamburger SV": "汉堡",
    "Juventus FC": "尤文图斯", "AC Milan": "AC米兰", "FC Internazionale Milano": "国际米兰",
    "SSC Napoli": "那不勒斯", "AS Roma": "罗马", "SS Lazio": "拉齐奥", "Atalanta BC": "亚特兰大",
    "ACF Fiorentina": "佛罗伦萨", "Torino FC": "都灵", "Bologna FC": "博洛尼亚",
    "Bologna FC 1909": "博洛尼亚", "Udinese Calcio": "乌迪内斯", "US Sassuolo Calcio": "萨索洛",
    "Hellas Verona FC": "维罗纳", "US Lecce": "莱切", "Cagliari Calcio": "卡利亚里",
    "Empoli FC": "恩波利", "Genoa CFC": "热那亚", "US Salernitana": "萨勒尼塔纳",
    "Frosinone Calcio": "弗罗西诺内", "AC Monza": "蒙扎", "Parma Calcio 1913": "帕尔马",
    "Como 1907": "科莫", "Venezia FC": "威尼斯",
    "Paris Saint-Germain FC": "巴黎圣日耳曼", "Paris Saint Germain FC": "巴黎圣日耳曼",
    "Olympique de Marseille": "马赛", "Olympique Lyonnais": "里昂", "AS Monaco FC": "摩纳哥",
    "LOSC Lille": "里尔", "Lille OSC": "里尔", "Stade Rennais FC": "雷恩",
    "Stade Rennais FC 1901": "雷恩", "OGC Nice": "尼斯", "RC Strasbourg Alsace": "斯特拉斯堡",
    "FC Nantes": "南特", "Montpellier HSC": "蒙彼利埃", "Stade Brestois 29": "布雷斯特",
    "Stade de Reims": "兰斯", "FC Lorient": "洛里昂", "Clermont Foot": "克莱蒙",
    "Toulouse FC": "图卢兹", "AJ Auxerre": "欧塞尔", "Angers SCO": "昂热", "FC Metz": "梅斯",
    "RC Lens": "朗斯", "Racing Club de Lens": "朗斯", "Havre AC": "勒阿弗尔",
    "Le Havre AC": "勒阿弗尔", "AS Saint-Étienne": "圣埃蒂安", "Stade Malherbe Caen": "卡昂",
    "FC Girondins de Bordeaux": "波尔多", "ESTAC Troyes": "特鲁瓦", "ES Troyes AC": "特鲁瓦",
    "Dijon FCO": "第戎", "Le Mans FC": "勒芒", "Paris FC": "巴黎FC",
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

def get_upcoming_matches(league_code, days_ahead=2):
    if not API_KEY:
        print(f"⚠️ 未设置FOOTBALL_DATA_API_KEY，跳过{league_code}赛程拉取")
        return []
    headers = {"X-Auth-Token": API_KEY}
    date_from = datetime.utcnow().strftime("%Y-%m-%d")
    date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED", "limit": 100}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        matches = response.json()["matches"]
        upcoming = []
        for m in matches:
            home_en = normalize_team_name(m["homeTeam"]["name"])
            away_en = normalize_team_name(m["awayTeam"]["name"])
            upcoming.append({"league": league_code, "home_team": home_en, "away_team": away_en, "date": m["utcDate"]})
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

def calc_handicap_probs(matrix, handicap_goals, favorite_is_home=True):
    win, draw, lose = 0.0, 0.0, 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if favorite_is_home:
                adjusted = i - handicap_goals
                opp = j
            else:
                adjusted = j - handicap_goals
                opp = i
            if handicap_goals in [0.5, 1.5]:
                if adjusted > opp:
                    win += matrix[i, j]
                else:
                    lose += matrix[i, j]
            else:
                if adjusted > opp:
                    win += matrix[i, j]
                elif adjusted == opp:
                    draw += matrix[i, j]
                else:
                    lose += matrix[i, j]
    return win, draw, lose

def match_probabilities(lambda_home, lambda_away, half_ratio=0.44):
    matrix = poisson_prob_matrix(lambda_home, lambda_away)
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    score_probs = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            score_probs[f"{i}-{j}"] = matrix[i,j]
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    lambda_half_home = lambda_home * half_ratio
    lambda_half_away = lambda_away * half_ratio
    half_matrix = poisson_prob_matrix(lambda_half_home, lambda_half_away, max_goals=4)
    ht_home = np.sum(np.tril(half_matrix, -1))
    ht_draw = np.sum(np.diag(half_matrix))
    ht_away = np.sum(np.triu(half_matrix, 1))
    
    htft_probs = {
        "胜胜": ht_home * home_win, "胜平": ht_home * draw, "胜负": ht_home * away_win,
        "平胜": ht_draw * home_win, "平平": ht_draw * draw, "平负": ht_draw * away_win,
        "负胜": ht_away * home_win, "负平": ht_away * draw, "负负": ht_away * away_win,
    }
    top_htft = sorted(htft_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # ===== 多档让球（0.5/1/1.5/2，去掉0球） =====
    favorite_is_home = lambda_home >= lambda_away
    handicap_levels = [0.5, 1.0, 1.5, 2.0]
    handicap_all = {}
    for level in handicap_levels:
        h_win, h_draw, h_lose = calc_handicap_probs(matrix, level, favorite_is_home)
        handicap_all[level] = (h_win, h_draw, h_lose)
    
    # 如果让0.5球强队赢盘概率<40%，说明双方实力接近，平手盘
    if handicap_all[0.5][0] < 0.40:
        handicap = {
            "favorite": "home" if favorite_is_home else "away",
            "handicap_goals": None,
            "is_draw_handicap": True,
            "win_prob": 0, "draw_prob": 0, "lose_prob": 0,
            "all_levels": handicap_all,
            "is_steady": False, "is_strong_steady": False,
        }
    else:
        best_level = min(handicap_levels, key=lambda l: abs(handicap_all[l][0] - 0.5))
        h_win, h_draw, h_lose = handicap_all[best_level]
        handicap = {
            "favorite": "home" if favorite_is_home else "away",
            "handicap_goals": best_level,
            "is_draw_handicap": False,
            "win_prob": h_win, "draw_prob": h_draw, "lose_prob": h_lose,
            "all_levels": handicap_all,
            "is_steady": h_win >= 0.45, "is_strong_steady": h_win >= 0.50,
        }
    
    total_goals = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total = i + j
            total_goals[total] = total_goals.get(total, 0) + matrix[i,j]
    sorted_total = sorted(total_goals.items(), key=lambda x: x[0])
    
    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "top_scores": top_scores, "top_htft": top_htft, "htft_probs": htft_probs,
        "handicap": handicap, "total_goals": sorted_total, "ht_draw_prob": ht_draw
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

def calculate_league_half_ratio(all_matches_df):
    ratios = {}
    default_ratio = 0.44
    for league in LEAGUES:
        if "league" in all_matches_df.columns:
            league_df = all_matches_df[all_matches_df["league"] == league]
        else:
            ratios[league] = default_ratio
            continue
        if "ht_home_goals" not in league_df.columns:
            ratios[league] = default_ratio
            print(f"⚠️ {league} 无半场数据列，使用默认值 {default_ratio}")
            continue
        valid = league_df.dropna(subset=["ht_home_goals", "ht_away_goals"])
        if len(valid) < 50:
            ratios[league] = default_ratio
            print(f"⚠️ {league} 半场数据不足50场（{len(valid)}场），使用默认值 {default_ratio}")
            continue
        total_goals = valid["home_goals"].sum() + valid["away_goals"].sum()
        ht_goals = valid["ht_home_goals"].sum() + valid["ht_away_goals"].sum()
        if total_goals > 0:
            ratio = ht_goals / total_goals
            ratios[league] = ratio
            ht_draw_count = len(valid[valid["ht_home_goals"] == valid["ht_away_goals"]])
            ht_draw_rate = ht_draw_count / len(valid)
            print(f"✅ {league} 半场进球比：{ratio:.3f}，半场平局率：{ht_draw_rate:.1%}（{len(valid)}场样本）")
        else:
            ratios[league] = default_ratio
    return ratios

def grade_match(fused_probs, odds_data):
    max_prob = max(fused_probs)
    dir_idx = np.argmax(fused_probs)
    dir_name = ["主胜", "平局", "客胜"][dir_idx]
    if max_prob >= 0.5:
        prob_grade = "🟢 绿灯"
        grade = "A档"
    elif max_prob >= 0.4:
        prob_grade = "🟡 黄灯"
        grade = "B档"
    elif max_prob >= 0.35:
        prob_grade = "🟠 橙灯"
        grade = "C档"
    else:
        prob_grade = "🔴 红灯"
        grade = "不推荐"
    odds_comment = "无赔率数据"
    final_grade = grade
    ev = None
    is_low_odds = False
    if odds_data:
        rec_odds = odds_data[dir_idx]
        ev = max_prob * rec_odds - 1
        if rec_odds <= 1.30:
            is_low_odds = True
            odds_comment = "⚠️ 赔率过低，无单场投注价值"
        elif 1.30 < rec_odds <= 1.50:
            odds_comment = "⚠️ 赔率偏低"
            if max_prob < 0.55 and grade == "A档":
                final_grade = "B档"
            elif grade in ["B档", "C档"]:
                final_grade = "C档" if grade == "B档" else "不推荐"
        elif 1.50 < rec_odds <= 2.20:
            odds_comment = "✅ 赔率合理"
        elif 2.20 < rec_odds <= 3.00:
            odds_comment = "⚠️ 赔率偏高"
            if max_prob < 0.50:
                final_grade = "C档" if grade == "B档" else "不推荐"
        else:
            odds_comment = "❌ 赔率超出合理区间"
            if max_prob < 0.60:
                final_grade = "不推荐"
    return {
        "direction": dir_name, "dir_idx": dir_idx, "max_prob": max_prob,
        "prob_grade": prob_grade, "grade": final_grade, "odds_comment": odds_comment,
        "ev": ev, "rec_odds": odds_data[dir_idx] if odds_data else None,
        "is_low_odds": is_low_odds
    }

def calculate_hf_combos(htft_probs, ht_draw_prob, full_dir_idx, odds_data=None):
    # 场类型判断
    if ht_draw_prob >= PING_STRICT_THRESHOLD:
        ping_type = "严格平系"
    elif ht_draw_prob >= PING_BALANCED_THRESHOLD:
        ping_type = "均衡平系"
    else:
        ping_type = "非平系"
    
    # SP估算：有赔率用赔率推导，无赔率用概率×0.85估算
    sp_est = {}
    sp_is_estimated = False
    if odds_data:
        h_odds, d_odds, a_odds = odds_data
        sp_est["平平"] = d_odds * 1.8
        sp_est["平胜"] = h_odds * 2.2
        sp_est["平负"] = a_odds * 2.2
        sp_est["胜胜"] = h_odds * 1.6
        sp_est["负负"] = a_odds * 1.6
        sp_est["胜平"] = d_odds * 1.5
        sp_est["负平"] = d_odds * 1.5
    else:
        sp_is_estimated = True
        for key in ["平平", "平胜", "平负", "胜胜", "负负", "胜平", "负平"]:
            prob = htft_probs.get(key, 0.01)
            sp_est[key] = (1 / prob) * 0.85 if prob > 0.01 else 50.0
    
    # 非平系场：按全场方向推荐，不硬推平平
    if ping_type == "非平系":
        if full_dir_idx == 0:  # 主胜
            primary_options = [("胜胜", htft_probs["胜胜"]), ("平胜", htft_probs["平胜"])]
            secondary_options = [("平平", htft_probs["平平"]), ("平胜", htft_probs["平胜"])]
        elif full_dir_idx == 2:  # 客胜
            primary_options = [("负负", htft_probs["负负"]), ("平负", htft_probs["平负"])]
            secondary_options = [("平平", htft_probs["平平"]), ("平负", htft_probs["平负"])]
        else:  # 平局
            primary_options = [("平平", htft_probs["平平"]), ("胜平", htft_probs["胜平"])]
            secondary_options = [("平平", htft_probs["平平"]), ("负平", htft_probs["负平"])]
        
        def build_combo(options):
            name1, prob1 = options[0]
            name2, prob2 = options[1]
            total_prob = prob1 + prob2
            sp1 = sp_est.get(name1, 0)
            sp2 = sp_est.get(name2, 0)
            ev = (prob1 * sp1 + prob2 * sp2) - 2
            ev_pct = ev / 2 * 100
            return {
                "combo": f"{name1} + {name2}",
                "total_prob": total_prob,
                "sp": f"{sp1:.2f} / {sp2:.2f}" + ("（估算）" if sp_is_estimated else ""),
                "sp_avg": (sp1 + sp2) / 2,
                "ev_pct": ev_pct,
            }
        
        first = build_combo(primary_options)
        second = build_combo(secondary_options)
        
        result = {
            "符合平系标准": False,
            "半场平概率": ht_draw_prob,
            "ping_type": ping_type,
            "阈值": PING_BALANCED_THRESHOLD,
        }
        result["首选"] = first
        result["次选"] = second
        result["博弈备选"] = None
        return result
    
    # 平系场（严格/均衡）：双选平平+平X
    ping_combos = {"平平": htft_probs["平平"], "平胜": htft_probs["平胜"], "平负": htft_probs["平负"]}
    combos = []
    options = list(ping_combos.items())
    for i in range(len(options)):
        for j in range(i+1, len(options)):
            name1, prob1 = options[i]
            name2, prob2 = options[j]
            total_prob = prob1 + prob2
            sp1 = sp_est.get(name1, 0)
            sp2 = sp_est.get(name2, 0)
            ev = (prob1 * sp1 + prob2 * sp2) - 2
            ev_pct = ev / 2 * 100
            align_score = 0
            if full_dir_idx == 0:
                if "平胜" in (name1, name2):
                    align_score = 100
            elif full_dir_idx == 2:
                if "平负" in (name1, name2):
                    align_score = 100
            else:
                if "平平" in (name1, name2):
                    align_score = 100
            combos.append({
                "combo": f"{name1} + {name2}", "total_prob": total_prob,
                "sp": f"{sp1:.2f} / {sp2:.2f}" + ("（估算）" if sp_is_estimated else ""),
                "sp_avg": (sp1 + sp2) / 2, "ev_pct": ev_pct, "align_score": align_score
            })
    
    first_candidates = [c for c in combos if c["align_score"] == 100]
    first_candidates.sort(key=lambda x: x["total_prob"], reverse=True)
    first = first_candidates[0] if first_candidates else None
    second_candidates = [c for c in combos if c != first]
    second_candidates.sort(key=lambda x: x["ev_pct"] if x["ev_pct"] is not None else -999, reverse=True)
    second = second_candidates[0] if second_candidates else None
    gamble = None
    for c in combos:
        if c == first or c == second:
            continue
        if c["ev_pct"] is not None and -2 <= c["ev_pct"] < 0:
            gamble = c
            break
    
    result = {
        "符合平系标准": ht_draw_prob >= PING_BALANCED_THRESHOLD,
        "半场平概率": ht_draw_prob,
        "ping_type": ping_type,
        "阈值": PING_STRICT_THRESHOLD,
    }
    result["首选"] = first
    result["次选"] = second
    result["博弈备选"] = gamble
    return result

def build_hf_parlay(ping_list, combo_type="首选"):
    if len(ping_list) < 2:
        return None
    candidates = []
    for i in range(len(ping_list)):
        for j in range(i+1, len(ping_list)):
            r1 = ping_list[i]
            r2 = ping_list[j]
            c1 = r1["hf_combos"].get(combo_type)
            c2 = r2["hf_combos"].get(combo_type)
            if not c1 or not c2:
                continue
            total_sp = c1["sp_avg"] * c2["sp_avg"]
            total_prob = c1["total_prob"] * c2["total_prob"]
            cross_league = r1["league"] != r2["league"]
            score = 0
            if cross_league:
                score += 100
            if 8 <= total_sp <= 15:
                score += 50
            elif 5 <= total_sp < 8:
                score += 20
            score += total_prob * 100
            candidates.append({
                "r1": r1, "r2": r2, "c1": c1, "c2": c2,
                "total_sp": total_sp, "total_prob": total_prob,
                "cross_league": cross_league, "score": score
            })
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

# ========== 战意系数模块（赛季过滤版） ==========
def calculate_league_standings(league_df, league):
    if league_df.empty:
        return {}, 0
    
    # 赛季过滤：当前赛季定义为每年7月1日至次年6月30日
    now = datetime.now()
    if now.month >= 7:
        season_start = datetime(now.year, 7, 1)
    else:
        season_start = datetime(now.year - 1, 7, 1)
    
    df = league_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= season_start]
    
    season_match_count = len(df)
    
    if "status" in df.columns:
        finished = df[df["status"].isin(["FINISHED", "finished", "已完赛"])]
    else:
        finished = df.copy()
    if finished.empty:
        return {}, season_match_count
    team_stats = {}
    for _, row in finished.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        hg = row.get("home_goals", 0)
        ag = row.get("away_goals", 0)
        if pd.isna(hg) or pd.isna(ag):
            continue
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {"points": 0, "gf": 0, "ga": 0, "played": 0}
        team_stats[home]["played"] += 1
        team_stats[away]["played"] += 1
        team_stats[home]["gf"] += hg
        team_stats[home]["ga"] += ag
        team_stats[away]["gf"] += ag
        team_stats[away]["ga"] += hg
        if hg > ag:
            team_stats[home]["points"] += 3
        elif hg == ag:
            team_stats[home]["points"] += 1
            team_stats[away]["points"] += 1
        else:
            team_stats[away]["points"] += 3
    for team in team_stats:
        team_stats[team]["gd"] = team_stats[team]["gf"] - team_stats[team]["ga"]
    sorted_teams = sorted(team_stats.items(), key=lambda x: (x[1]["points"], x[1]["gd"]), reverse=True)
    for rank, (team, _) in enumerate(sorted_teams, 1):
        team_stats[team]["rank"] = rank
    return team_stats, season_match_count

def get_current_round(standings, league):
    if not standings:
        return 1
    avg_played = np.mean([s["played"] for s in standings.values()])
    return max(1, round(avg_played))

def get_motivation(team, standings, league, current_round, season_data_sufficient=True):
    if team not in standings:
        return "正常", 0, "无积分榜数据"
    info = standings[team]
    rank = info["rank"]
    points = info["points"]
    total_teams = LEAGUE_TEAM_COUNT.get(league, 20)
    total_rounds = LEAGUE_TOTAL_ROUNDS.get(league, 38)
    if current_round <= 5:
        return "正常", 0, "赛季初，战意参考价值低"
    if not season_data_sufficient:
        return "正常", 0, "赛季数据不足，战意参考价值低"
    sorted_by_points = sorted(standings.items(), key=lambda x: x[1]["points"], reverse=True)
    top_points = sorted_by_points[0][1]["points"] if sorted_by_points else 0
    if rank <= 4 and (top_points - points) <= 5:
        return "强", MOTIVATION_SCORE["强"], f"争冠组（排名第{rank}，距榜首{top_points-points}分）"
    europa_line = LEAGUE_EUROPA_LINE.get(league, 6)
    if europa_line - 2 <= rank <= europa_line + 2:
        if rank <= europa_line:
            return "强", MOTIVATION_SCORE["强"], f"欧战区内（排名第{rank}）"
        else:
            line_points = sorted_by_points[europa_line - 1][1]["points"] if len(sorted_by_points) >= europa_line else 0
            diff = line_points - points
            if diff <= 5:
                return "强", MOTIVATION_SCORE["强"], f"冲击欧战（排名第{rank}，距欧战区{diff}分）"
    relegation_zone_start = total_teams - 2
    safety_rank = total_teams - 3
    if rank >= relegation_zone_start:
        return "强", MOTIVATION_SCORE["强"], f"保级区（排名第{rank}）"
    if safety_rank <= rank <= total_teams - 4 + 1:
        safety_points = sorted_by_points[safety_rank - 1][1]["points"] if len(sorted_by_points) >= safety_rank else 0
        diff = safety_points - points
        if diff <= 3:
            return "强", MOTIVATION_SCORE["强"], f"保级边缘（排名第{rank}，距安全线{diff}分）"
    if current_round >= total_rounds - 4:
        if 8 <= rank <= total_teams - 6:
            if rank > europa_line + 2 and rank < safety_rank - 1:
                return "弱", MOTIVATION_SCORE["弱"], "赛季末无欲无求"
    return "正常", 0, "中游正常战意"

def apply_motivation_adjustment(home_team, away_team, fused_probs, standings, league, current_round, season_data_sufficient=True):
    home_level, home_score, home_desc = get_motivation(home_team, standings, league, current_round, season_data_sufficient)
    away_level, away_score, away_desc = get_motivation(away_team, standings, league, current_round, season_data_sufficient)
    motivation_diff = home_score - away_score
    if motivation_diff == 0:
        return fused_probs.copy(), "双方战意相当，不修正", home_level, away_level
    total_rounds = LEAGUE_TOTAL_ROUNDS.get(league, 38)
    if current_round <= 5:
        season_factor = 0.3
    elif current_round >= total_rounds - 4:
        season_factor = 1.5
    else:
        season_factor = 1.0
    if current_round < total_rounds * 0.5:
        season_factor *= 0.5
    if not season_data_sufficient:
        season_factor *= 0.5
    abs_diff = abs(motivation_diff)
    if abs_diff >= 25:
        base_home_factor = 1.05 if motivation_diff > 0 else 0.95
        base_away_factor = 0.95 if motivation_diff > 0 else 1.05
        draw_factor = 0.98
    elif abs_diff >= 10:
        base_home_factor = 1.03 if motivation_diff > 0 else 0.97
        base_away_factor = 0.97 if motivation_diff > 0 else 1.03
        draw_factor = 1.0
    else:
        return fused_probs.copy(), "战意差异较小，不修正", home_level, away_level
    home_factor = 1.0 + (base_home_factor - 1.0) * season_factor
    away_factor = 1.0 + (base_away_factor - 1.0) * season_factor
    draw_factor_adj = 1.0 + (draw_factor - 1.0) * season_factor
    adjusted = [fused_probs[0] * home_factor, fused_probs[1] * draw_factor_adj, fused_probs[2] * away_factor]
    total = sum(adjusted)
    adjusted = [p / total for p in adjusted]
    max_change = max(abs(adjusted[i] - fused_probs[i]) for i in range(3))
    if max_change > 0.03:
        scale = 0.03 / max_change
        adjusted = [fused_probs[i] + (adjusted[i] - fused_probs[i]) * scale for i in range(3)]
        total = sum(adjusted)
        adjusted = [p / total for p in adjusted]
    home_change = (adjusted[0] - fused_probs[0]) * 100
    if home_change > 0:
        desc = f"主队{home_level}战意，客队{away_level}战意，主胜概率上调{home_change:.1f}%"
    else:
        desc = f"主队{home_level}战意，客队{away_level}战意，主胜概率下调{abs(home_change):.1f}%"
    return adjusted, desc, home_level, away_level

# ========== 主报告生成函数 ==========
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
    all_matches_df = pd.read_csv("data/matches.csv")
    print(f"📚 历史数据总场数：{len(all_matches_df)}")
    league_half_ratios = calculate_league_half_ratio(all_matches_df)
    elo_cache = {}
    league_df_cache = {}
    data_sufficient_cache = {}
    standings_cache = {}
    current_round_cache = {}
    season_match_count_cache = {}
    for league in LEAGUES:
        if "league" in all_matches_df.columns:
            league_df = all_matches_df[all_matches_df["league"] == league].copy()
        else:
            league_df = all_matches_df.copy() if league == "PL" else pd.DataFrame()
        league_df_cache[league] = league_df
        elo_cache[league] = load_or_calculate_elo(league_df, league)
        data_sufficient_cache[league] = len(league_df) >= 10
        standings, season_count = calculate_league_standings(league_df, league)
        standings_cache[league] = standings
        season_match_count_cache[league] = season_count
        current_round_cache[league] = get_current_round(standings, league)
        season_data_sufficient = season_count >= 50
        if standings:
            print(f"📊 {LEAGUE_NAMES.get(league, league)} 当前赛季已完赛{season_count}场，积分榜{len(standings)}队，约第{current_round_cache[league]}轮，数据{'充足' if season_data_sufficient else '不足'}")
    
    match_results = []
    match_counter = 1
    all_upcoming_sorted = sorted(all_upcoming, key=lambda x: x["date"])
    for match in all_upcoming_sorted:
        league = match["league"]
        league_df = league_df_cache.get(league, pd.DataFrame())
        elo_dict = elo_cache.get(league, {})
        data_sufficient = data_sufficient_cache.get(league, False)
        league_name = LEAGUE_NAMES.get(league, league)
        params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS["PL"])
        standings = standings_cache.get(league, {})
        current_round = current_round_cache.get(league, 1)
        season_data_sufficient = season_match_count_cache.get(league, 0) >= 50
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
            half_ratio = league_half_ratios.get(league, 0.44)
            poisson_probs = match_probabilities(lh, la, half_ratio=half_ratio)
            p_poisson_raw = [poisson_probs["home_win"], poisson_probs["draw"], poisson_probs["away_win"]]
            p_poisson = [p_poisson_raw[0], p_poisson_raw[1] * params["draw_corr"], p_poisson_raw[2]]
            total_p = sum(p_poisson)
            p_poisson = [p / total_p for p in p_poisson]
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
            fused_probs_raw = [fused_home, fused_draw, fused_away]
            fused_probs, motivation_desc, home_motivation, away_motivation = apply_motivation_adjustment(
                home_en, away_en, fused_probs_raw, standings, league, current_round, season_data_sufficient
            )
            grade_info = grade_match(fused_probs, odds_data)
            full_dir_idx = grade_info["dir_idx"]
            handicap = poisson_probs["handicap"]
            if handicap["favorite"] == "home":
                favorite_name = home_zh
                poisson_fav_win = poisson_probs["home_win"]
                fused_fav_win = fused_probs[0]
            else:
                favorite_name = away_zh
                poisson_fav_win = poisson_probs["away_win"]
                fused_fav_win = fused_probs[2]
            handicap["favorite_name"] = favorite_name
            
            # 让球盘概率按融合概率校准
            if not handicap.get("is_draw_handicap") and poisson_fav_win > 0:
                calibrate_factor = fused_fav_win / poisson_fav_win
                calibrated_win = handicap["win_prob"] * calibrate_factor
                if 0 < calibrated_win < 1:
                    remaining = 1 - calibrated_win
                    total_other = handicap["draw_prob"] + handicap["lose_prob"]
                    if total_other > 0:
                        handicap["draw_prob"] = handicap["draw_prob"] / total_other * remaining
                        handicap["lose_prob"] = handicap["lose_prob"] / total_other * remaining
                        handicap["win_prob"] = calibrated_win
                        handicap["is_steady"] = handicap["win_prob"] >= 0.45
                        handicap["is_strong_steady"] = handicap["win_prob"] >= 0.50
            
            hf_combos = calculate_hf_combos(poisson_probs["htft_probs"], poisson_probs["ht_draw_prob"], full_dir_idx, odds_data)
            top_ttg = sorted(poisson_probs["total_goals"], key=lambda x: x[1], reverse=True)[:3]
            divergence = None
            if p_market:
                diff = abs(fused_probs[grade_info["dir_idx"]] - p_market[grade_info["dir_idx"]])
                divergence = {
                    "value": diff, "is_large": diff >= 0.1,
                    "type": "方向对立" if np.argmax(fused_probs) != np.argmax(p_market) else "价值追击"
                }
            dirs = [np.argmax(p_poisson), np.argmax(p_elo)]
            if p_market:
                dirs.append(np.argmax(p_market))
            same_count = sum(1 for d in dirs if d == dirs[0])
            model_consistent = same_count == len(dirs)
            match_date_str = match_dt.strftime("%Y-%m-%d")
            match_id = generate_match_id(match_date_str, home_en, away_en)
            result = {
                "match_no": f"{match_counter:03d}", "match_id": match_id,
                "league": league, "league_name": league_name,
                "home_zh": home_zh, "away_zh": away_zh, "match_time": match_time,
                "lh": lh, "la": la, "p_poisson": p_poisson, "p_market": p_market, "p_elo": p_elo,
                "fused_probs": fused_probs, "fused_probs_raw": fused_probs_raw,
                "grade_info": grade_info, "handicap": handicap, "hf_combos": hf_combos,
                "top_scores": poisson_probs["top_scores"], "top_ttg": top_ttg,
                "divergence": divergence, "model_consistent": model_consistent,
                "odds_data": odds_data, "data_sufficient": data_sufficient,
                "motivation_desc": motivation_desc, "home_motivation": home_motivation, "away_motivation": away_motivation
            }
            match_results.append(result)
            match_counter += 1
        except Exception as e:
            print(f"❌ 跳过比赛 {home_en} vs {away_en}：{str(e)}")
            traceback.print_exc()
            continue
    
    now_bj = datetime.utcnow() + timedelta(hours=8)
    report = ""
    report += "# 足球预测报告（全面修复版）\n\n"
    report += f"**生成时间**：{now_bj.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n"
    report += f"**预测时段**：{TARGET_DATE_LABEL} 17:00 ~ 次日 12:00\n"
    report += f"**数据来源**：Football-Data.org + The Odds API + ELO\n"
    has_any_odds = any(r["odds_data"] for r in match_results)
    if has_any_odds:
        model_desc = "泊松模型（DC修正+联赛校准） + 全庄家赔率中位数 + ELO + 战意系数（赛季过滤） + 多档让球（0.5/1/1.5/2） + 分级平系策略"
    else:
        model_desc = "泊松模型（DC修正+联赛校准） + ELO + 战意系数（赛季过滤） + 多档让球 + 分级平系策略（本期无赔率数据）"
    report += f"**模型说明**：{model_desc}\n\n"
    report += "> ⚠️ 风险提示：所有预测仅供参考，不构成投注建议。足球比赛不确定性高，请理性购彩，量力而行。\n\n"
    
    # EV说明
    report += "> 📊 **EV参考说明**：足球博彩中绝大多数选项EV为负（庄家抽水约10%-15%），属正常现象。EV参考标准：≥0% 价值高，-5%~0% 价值正常，-15%~-5% 价值偏低，<-15% 价值低不建议介入。EV仅作辅助参考，核心看概率与赔率的平衡。\n\n"
    # SP说明
    report += "> 💰 **SP赔率说明**：有真实赔率的场次基于赔率推导，无真实赔率的场次用模型概率估算（×0.85抽水系数），均标注「估算」，仅供参考。\n\n"
    
    leagues_covered = set(r["league_name"] for r in match_results)
    ping_strict_count = sum(1 for r in match_results if r["hf_combos"].get("ping_type") == "严格平系")
    ping_balanced_count = sum(1 for r in match_results if r["hf_combos"].get("ping_type") == "均衡平系")
    grade_a_count = sum(1 for r in match_results if r["grade_info"]["grade"] == "A档")
    grade_b_count = sum(1 for r in match_results if r["grade_info"]["grade"] == "B档")
    motivation_adjusted = sum(1 for r in match_results if "不修正" not in r["motivation_desc"] and "较小" not in r["motivation_desc"])
    steady_count = sum(1 for r in match_results if r["handicap"]["is_steady"] and not r["handicap"].get("is_draw_handicap"))
    strong_steady_count = sum(1 for r in match_results if r["handicap"]["is_strong_steady"] and not r["handicap"].get("is_draw_handicap"))
    draw_handicap_count = sum(1 for r in match_results if r["handicap"].get("is_draw_handicap"))
    
    report += f"**本期概览**：共 {len(match_results)} 场比赛，覆盖 {'、'.join(leagues_covered)}；\n"
    report += f"- A档强推：{grade_a_count} 场 | B档可买：{grade_b_count} 场\n"
    report += f"- 平系策略：严格场 {ping_strict_count} 场 | 均衡场 {ping_balanced_count} 场\n"
    report += f"- 让球稳胆：{steady_count} 场（强稳胆{strong_steady_count}场） | 平手盘无让球参考：{draw_handicap_count} 场\n"
    report += f"- 战意修正场次：{motivation_adjusted} 场（赛季过滤后）\n\n"
    report += "---\n\n"
    
    # ===== 今日最终推荐 =====
    report += "## 🎯 今日最终推荐（直接看这里）\n\n"
    
    # 稳健二串一推荐
    report += "### 一、稳健二串一推荐\n\n"
    ab_grade_for_parlay = [r for r in match_results 
                           if r["grade_info"]["grade"] in ["A档", "B档"] 
                           and r["odds_data"] 
                           and r["grade_info"]["rec_odds"] >= 1.40
                           and not r["grade_info"]["is_low_odds"]]
    
    parlay_combos = []
    for i in range(len(ab_grade_for_parlay)):
        for j in range(i+1, len(ab_grade_for_parlay)):
            r1 = ab_grade_for_parlay[i]
            r2 = ab_grade_for_parlay[j]
            total_odds = r1["grade_info"]["rec_odds"] * r2["grade_info"]["rec_odds"]
            total_prob = r1["grade_info"]["max_prob"] * r2["grade_info"]["max_prob"]
            # 评分：赔率在2.4-3.2最优，跨联赛加分，命中率加分
            score = 0
            if 2.4 <= total_odds <= 3.2:
                score += 100
            elif 2.2 <= total_odds < 2.4 or 3.2 < total_odds <= 3.5:
                score += 60
            else:
                score += 20
            if r1["league"] != r2["league"]:
                score += 50
            score += total_prob * 50
            parlay_combos.append({
                "r1": r1, "r2": r2, "total_odds": total_odds,
                "total_prob": total_prob, "score": score
            })
    
    parlay_combos.sort(key=lambda x: x["score"], reverse=True)
    top_parlays = parlay_combos[:3]
    
    if top_parlays:
        report += "| 组合 | 对阵1 | 选项 | 档位 | 赔率 | 对阵2 | 选项 | 档位 | 赔率 | 总赔率 | 综合命中率 | 推荐理由 |\n"
        report += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        for idx, pc in enumerate(top_parlays, 1):
            r1 = pc["r1"]
            r2 = pc["r2"]
            reasons = []
            if r1["league"] != r2["league"]:
                reasons.append("跨联赛")
            if 2.4 <= pc["total_odds"] <= 3.2:
                reasons.append("赔率黄金区间")
            elif 2.2 <= pc["total_odds"] <= 3.5:
                reasons.append("赔率合理区间")
            if r1["model_consistent"] and r2["model_consistent"]:
                reasons.append("双场模型一致")
            reason_str = "、".join(reasons)
            report += f"| 组合{idx} | {r1['home_zh']}vs{r1['away_zh']} | {r1['grade_info']['direction']} | {r1['grade_info']['grade']} | {r1['grade_info']['rec_odds']:.2f} | {r2['home_zh']}vs{r2['away_zh']} | {r2['grade_info']['direction']} | {r2['grade_info']['grade']} | {r2['grade_info']['rec_odds']:.2f} | {pc['total_odds']:.2f} | 约{pc['total_prob']:.0%} | {reason_str} |\n"
        report += "\n"
    else:
        report += "> 📌 今日无符合赔率要求（单场≥1.40）的稳健串关组合，建议单场投注或休息。\n\n"
    
    # 平系半全场推荐
    report += "### 二、平系半全场推荐（平平+平X双选）\n\n"
    ping_candidates = [r for r in match_results 
                      if r["hf_combos"].get("ping_type") in ["严格平系", "均衡平系"]
                      and r["hf_combos"]["首选"]]
    
    ping_candidates.sort(key=lambda r: r["hf_combos"]["首选"]["total_prob"] * r["hf_combos"]["首选"]["sp_avg"], reverse=True)
    top_ping = ping_candidates[:3]
    
    if top_ping:
        report += "| 编号 | 对阵 | 组合 | 半场平概率 | 场类型 | 双选概率 | 估算SP | 期望收益 | 推荐理由 |\n"
        report += "|---|---|---|---|---|---|---|---|---|\n"
        for idx, r in enumerate(top_ping, 1):
            c = r["hf_combos"]["首选"]
            ping_type = r["hf_combos"].get("ping_type", "")
            reasons = []
            reasons.append(f"半场平{r['hf_combos']['半场平概率']:.0%}")
            dir_name = r["grade_info"]["direction"]
            if "平胜" in c["combo"] and dir_name == "主胜":
                reasons.append("主胜方向对齐")
            elif "平负" in c["combo"] and dir_name == "客胜":
                reasons.append("客胜方向对齐")
            if c["ev_pct"] is not None and c["ev_pct"] > -10:
                reasons.append(f"EV{c['ev_pct']:+.0f}%")
            reason_str = "、".join(reasons)
            ev_str = f"{c['ev_pct']:+.1f}%" if c["ev_pct"] is not None else "-"
            report += f"| H{idx} | {r['home_zh']}vs{r['away_zh']} | {c['combo']} | {r['hf_combos']['半场平概率']:.1%} | {ping_type} | {c['total_prob']:.1%} | {c['sp_avg']:.2f} | {ev_str} | {reason_str} |\n"
        report += "\n"
        report += "> 💡 平系双选说明：单场双选概率约30%-40%，通过严格择赛（半场平概率≥44%）提升长期命中率。严格场置信度更高，均衡场次之。\n\n"
    else:
        report += "> 📌 今日无符合平系标准（半场平≥44%）的半全场推荐。\n\n"
    
    report += "---\n\n"
    
    # 总览汇总表
    report += "## 📋 总览汇总表\n\n"
    report += "💡 快速用法：直接筛「A档」比赛做主投，「B档」做串关，C档直接跳过。\n\n"
    report += "| 编号 | 对阵 | 胜平负首选 | 档位 | 让球参考 | 平系类型 | 半全场首选 | 总进球首选 |\n"
    report += "|------|------|------------|------|----------|----------|------------|------------|\n"
    for r in match_results:
        no = r["match_no"]
        vs = f"{r['home_zh']} vs {r['away_zh']}"
        first_dir = r["grade_info"]["direction"]
        first_prob = f"{r['grade_info']['max_prob']:.1%}"
        grade = r["grade_info"]["grade"]
        if r["grade_info"]["is_low_odds"]:
            grade += "（赔率低）"
        h = r["handicap"]
        if h.get("is_draw_handicap"):
            hf_ref = "平手盘，无让球参考"
        elif h["is_strong_steady"]:
            hf_ref = f"{h['favorite_name']}让{h['handicap_goals']:g}球 强稳"
        elif h["is_steady"]:
            hf_ref = f"{h['favorite_name']}让{h['handicap_goals']:g}球 稳"
        elif h["win_prob"] >= 0.40:
            hf_ref = f"让{h['handicap_goals']:g}球 准稳"
        else:
            hf_ref = f"让{h['handicap_goals']:g}球 价值低"
        ping_type = r["hf_combos"].get("ping_type", "")
        if r["hf_combos"]["首选"]:
            hf_first = r["hf_combos"]["首选"]["combo"]
        else:
            hf_first = "无推荐"
        ttg_first = f"{r['top_ttg'][0][0]}球" if r["top_ttg"] else "-"
        report += f"| {no} | {vs} | {first_dir} {first_prob} | {grade} | {hf_ref} | {ping_type} | {hf_first} | {ttg_first} |\n"
    report += "\n---\n\n"
    
    # 单场深度分析
    report += "## ⚽ 单场深度决策分析\n\n"
    current_league = ""
    for r in match_results:
        if r["league_name"] != current_league:
            current_league = r["league_name"]
            if r["data_sufficient"]:
                report += f"### {current_league}\n\n"
            else:
                report += f"### {current_league} ⚠️ 数据不足，仅供参考\n\n"
        report += f"#### {r['match_no']} {r['home_zh']} vs {r['away_zh']}\n\n"
        report += f"- **开赛时间**：{r['match_time']}\n"
        report += f"- **期望进球**：主 {r['lh']:.2f}，客 {r['la']:.2f}\n\n"
        
        report += "##### 一、单场胜平负（主力投注）\n\n"
        report += "**融合最终概率（含战意修正）**\n"
        report += "| 主胜 | 平局 | 客胜 |\n|---|---:|---:|\n"
        report += f"| {r['fused_probs'][0]:.1%} | {r['fused_probs'][1]:.1%} | {r['fused_probs'][2]:.1%} |\n\n"
        report += f"> ⚔️ **战意提示**：{r['motivation_desc']}\n"
        if "不修正" not in r["motivation_desc"] and "较小" not in r["motivation_desc"]:
            raw = r["fused_probs_raw"]
            report += f">   修正前：主胜{raw[0]:.1%} / 平局{raw[1]:.1%} / 客胜{raw[2]:.1%}\n"
        report += "\n"
        g = r["grade_info"]
        report += f"**概率档位**：{g['prob_grade']}（{g['direction']} {g['max_prob']:.1%}）\n"
        if g["rec_odds"]:
            report += f"**对应赔率**：{g['rec_odds']:.2f}\n"
            report += f"**赔率检查**：{g['odds_comment']}\n"
        report += f"**最终评级**：{g['grade']}\n"
        report += "> 📌 **判定依据**："
        reason_parts = []
        if g["max_prob"] >= 0.5:
            reason_parts.append("首选方向概率≥50%阈值")
        elif g["max_prob"] >= 0.4:
            reason_parts.append("首选方向概率≥40%阈值")
        if r["model_consistent"]:
            reason_parts.append("泊松/ELO/市场三模型方向一致")
        else:
            reason_parts.append("模型存在分歧，已降档处理")
        if g["is_low_odds"]:
            reason_parts.append("赔率过低，无单场投注价值，仅适合串关打底")
        elif g["rec_odds"] and 1.5 <= g["rec_odds"] <= 2.2:
            reason_parts.append("赔率处于合理区间")
        elif g["rec_odds"] and g["rec_odds"] < 1.5:
            reason_parts.append("赔率偏低，已做降档校验")
        report += "、".join(reason_parts) + "。\n"
        if g["ev"] is not None:
            ev_sign = "正收益" if g["ev"] > 0 else "负收益"
            report += f"> 💡 **EV解读**：{g['ev']*100:+.1f}%（长期重复买入预期{ev_sign}，{abs(g['ev']*100):.1f}%/百本金）\n"
        report += "\n**建议仓位**："
        if g["is_low_odds"]:
            report += "仅作串关稳胆，不建议单场投注"
        elif g["grade"] == "A档":
            report += "主力仓位（占总资金30%-50%）"
        elif g["grade"] == "B档":
            report += "轻仓（占总资金10%-20%）"
        elif g["grade"] == "C档":
            report += "娱乐小注（≤总资金5%）"
        else:
            report += "不建议介入"
        report += "\n\n"
        
        # 二、让球盘
        h = r["handicap"]
        if h.get("is_draw_handicap"):
            report += "##### 二、让球盘\n\n"
            report += "> 📌 双方实力接近（让0.5球强队赢盘概率<40%），真实盘口应为平手盘，无合适让球参考。建议直接参考胜平负。\n\n"
        else:
            handicap_text = f"{h['handicap_goals']:g}"
            report += f"##### 二、让球盘（串关稳胆参考 · {h['favorite_name']}让{handicap_text}球）\n\n"
            report += "> 📌 模型自动匹配盘口（强队赢盘概率最接近50%的档位），已按融合概率校准，真实盘口以官方为准。\n\n"
            report += "| 强队赢盘 | 走盘 | 弱队赢盘 |\n|---|---:|---:|\n"
            if h["draw_prob"] > 0.001:
                report += f"| {h['win_prob']:.1%} | {h['draw_prob']:.1%} | {h['lose_prob']:.1%} |\n\n"
            else:
                report += f"| {h['win_prob']:.1%} | - | {h['lose_prob']:.1%} |\n\n"
            if h["is_strong_steady"]:
                steady_text = f"✅✅ 强稳胆，{h['favorite_name']}让{handicap_text}球后赢盘概率超50%，可重仓做串关打底"
            elif h["is_steady"]:
                steady_text = f"✅ 稳胆，{h['favorite_name']}让{handicap_text}球后赢盘概率超45%，可作串关打底"
            elif h["win_prob"] >= 0.40:
                steady_text = f"⚠️ 准稳胆，赢盘概率{h['win_prob']:.0%}（40%~45%），轻仓参考，注意走盘风险"
            elif h["lose_prob"] >= 0.45:
                steady_text = f"❌ 不推荐，{h['favorite_name']}让{handicap_text}球后赢盘概率仅{h['win_prob']:.0%}，弱队赢盘概率高达{h['lose_prob']:.0%}，让球价值低"
            else:
                steady_text = f"⚠️ 赢盘概率{h['win_prob']:.0%}未达稳胆线，不建议单独做稳胆"
            report += f"**稳胆判定**：{steady_text}\n\n"
        
        # 三、半全场双选
        report += "##### 三、半全场双选\n\n"
        hf = r["hf_combos"]
        ping_type = hf.get("ping_type", "非平系")
        
        if ping_type == "严格平系":
            report += f"✅ **本场符合严格平系标准**\n"
            report += f"> 📌 判定依据：半场平局概率 {hf['半场平概率']:.1%}，≥ {PING_STRICT_THRESHOLD*100:.0f}%，高置信平系场，双选容错率强。\n\n"
        elif ping_type == "均衡平系":
            report += f"✅ **本场符合均衡平系标准**\n"
            report += f"> 📌 判定依据：半场平局概率 {hf['半场平概率']:.1%}，≥ {PING_BALANCED_THRESHOLD*100:.0f}%，中等置信平系场，双选容错率较好。\n\n"
        else:
            report += f"⚠️ **本场非平系场**\n"
            report += f"> 📌 判定依据：半场平局概率 {hf['半场平概率']:.1%}，低于 {PING_BALANCED_THRESHOLD*100:.0f}%，上半场分胜负概率更高。以下按全场方向推荐，仅作参考，不宜重仓。\n\n"
        
        if hf["首选"]:
            c = hf["首选"]
            report += "###### ✅ 首选组合\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **综合中奖概率**：{c['total_prob']:.1%}（约每{1/c['total_prob']:.1f}单中1单）\n"
            report += f"- **对应SP赔率**：{c['sp']}\n"
            if c["ev_pct"] is not None:
                report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        if hf["次选"]:
            c = hf["次选"]
            report += "###### ⚠️ 次选组合（收益增强）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **综合中奖概率**：{c['total_prob']:.1%}\n"
            report += f"- **对应SP赔率**：{c['sp']}\n"
            if c["ev_pct"] is not None:
                report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        if hf.get("博弈备选"):
            c = hf["博弈备选"]
            report += "###### 🎲 博弈备选（娱乐小注）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **对应SP赔率**：{c['sp']}\n"
            report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        
        # 四、总进球&比分
        report += "##### 四、总进球 & 比分参考\n\n"
        report += "**总进球TOP3**："
        ttg_str = [f"{x[0]}球({x[1]:.1%})" for x in r["top_ttg"]]
        report += "、".join(ttg_str) + "\n"
        report += "**比分TOP3**："
        score_str = [f"{x[0]}({x[1]:.1%})" for x in r["top_scores"]]
        report += "、".join(score_str) + "\n\n"
        
        # 五、模型vs市场分歧
        if r["divergence"]:
            report += "##### 五、模型 vs 市场分歧\n\n"
            div = r["divergence"]
            if div["is_large"]:
                report += f"⚠️ **分歧较大**：分歧值 {div['value']:.1%}，类型：{div['type']}\n"
                report += "> 🚨 风险提示：模型与市场预期差异大，存在搏冷或追热风险，谨慎介入，不建议重仓。\n"
            else:
                report += f"✅ 分歧较小：分歧值 {div['value']:.1%}，模型与市场预期基本一致。\n"
            report += "\n"
        
        # 六、综合决策建议
        report += "##### 六、综合决策建议（大白话）\n\n"
        if g["is_low_odds"]:
            main_advice = f"主力方向：{g['direction']}，概率{g['max_prob']:.0%}，但赔率{g['rec_odds']:.2f}过低，仅作串关稳胆，不建议单场投注"
        elif g["grade"] == "A档":
            main_advice = f"主力方向：{g['direction']}，概率{g['max_prob']:.0%}，可单场主力或做串关稳胆"
        elif g["grade"] == "B档":
            main_advice = f"主力方向：{g['direction']}，概率{g['max_prob']:.0%}，轻仓投注或串关搭配"
        elif g["grade"] == "C档":
            main_advice = f"方向：{g['direction']}，概率偏低，娱乐小注即可，不建议重仓"
        else:
            main_advice = "本场不建议介入，跳过"
        
        if h.get("is_draw_handicap"):
            handicap_advice = "让球：双方实力接近，平手盘，无让球参考，直接看胜平负"
        elif h["is_strong_steady"]:
            handicap_advice = f"让球：{h['favorite_name']}让{h['handicap_goals']:g}球强稳胆，赢盘概率{h['win_prob']:.0%}，可做串关打底"
        elif h["is_steady"]:
            handicap_advice = f"让球：{h['favorite_name']}让{h['handicap_goals']:g}球稳胆，赢盘概率{h['win_prob']:.0%}，可做串关打底"
        elif h["win_prob"] >= 0.40:
            handicap_advice = f"让球：{h['favorite_name']}让{h['handicap_goals']:g}球赢盘概率{h['win_prob']:.0%}，准稳胆，轻仓参考"
        else:
            handicap_advice = f"让球：{h['favorite_name']}让{h['handicap_goals']:g}球价值低，不建议碰让球盘"
        
        if ping_type in ["严格平系", "均衡平系"] and hf["首选"]:
            combo = hf["首选"]["combo"]
            prob = hf["首选"]["total_prob"]
            hf_advice = f"半全场：{ping_type}，双选{combo}，综合概率{prob:.0%}，{'主力推荐' if ping_type == '严格平系' else '次选推荐'}"
        else:
            hf_advice = "半全场：非平系场，按全场方向推荐，仅作参考，不宜重仓"
        
        ttg_top = r["top_ttg"][0][0] if r["top_ttg"] else 2
        ttg_second = r["top_ttg"][1][0] if len(r["top_ttg"]) > 1 else ttg_top + 1
        ttg_advice = f"总进球：首选{ttg_top}球，次选{ttg_second}球"
        
        if g["grade"] in ["A档", "B档"] and not g["is_low_odds"]:
            summary = "💡 一句话：本场可买，按上述仓位配置即可"
        elif g["is_low_odds"]:
            summary = "💡 一句话：概率高但赔率低，只适合做串关稳胆，不建议单场买"
        else:
            summary = "💡 一句话：本场风险较高，建议跳过或纯娱乐小注"
        
        report += f"- {main_advice}\n"
        report += f"- {handicap_advice}\n"
        report += f"- {hf_advice}\n"
        report += f"- {ttg_advice}\n"
        report += f"\n{summary}\n\n"
        
        report += "---\n\n"
    
    # 专题汇总
    report += "## 🎯 专题汇总\n\n"
    report += "### （1）搏冷备选汇总\n\n"
    cold_list = []
    for r in match_results:
        if r["divergence"] and r["divergence"]["is_large"]:
            cold_list.append(r)
    if cold_list:
        report += "| 对阵 | 搏冷方向 | 模型概率 | 市场概率 | 分歧值 | 类型 |\n"
        report += "|------|----------|----------|----------|--------|------|\n"
        for r in cold_list:
            vs = f"{r['home_zh']} vs {r['away_zh']}"
            diffs = [abs(r["fused_probs"][i] - r["p_market"][i]) for i in range(3)]
            cold_idx = np.argmax(diffs)
            cold_dir = ["主胜", "平局", "客胜"][cold_idx]
            model_p = r["fused_probs"][cold_idx]
            market_p = r["p_market"][cold_idx]
            div_v = diffs[cold_idx]
            div_type = r["divergence"]["type"]
            report += f"| {vs} | {cold_dir} | {model_p:.1%} | {market_p:.1%} | {div_v:.1%} | {div_type} |\n"
        report += "\n> 说明：仅做参考，不建议重仓；分歧越大不确定性越高。\n\n"
    else:
        report += "本期无明显分歧场次。\n\n"
    
    report += "### （2）市场过热提示\n\n"
    hot_list = []
    for r in match_results:
        if r["p_market"] and max(r["p_market"]) >= 0.65:
            hot_list.append(r)
    if hot_list:
        report += "以下场次热门方向市场支持率≥65%，赔率价值被压缩，追热性价比低，注意防冷：\n\n"
        for r in hot_list:
            hot_idx = np.argmax(r["p_market"])
            hot_dir = ["主胜", "平局", "客胜"][hot_idx]
            report += f"- **{r['home_zh']} vs {r['away_zh']}**：{hot_dir}市场概率 {r['p_market'][hot_idx]:.1%}\n"
        report += "\n"
    else:
        report += "本期无明显过热场次。\n\n"
    report += "---\n\n"
    
    # 分档位串关方案
    report += "## 🔗 分档位串关方案\n\n"
    ab_grade = [r for r in match_results if r["grade_info"]["grade"] in ["A档", "B档"] and r["odds_data"]]
    steady_handicap = [r for r in match_results if r["handicap"]["is_steady"] and not r["handicap"].get("is_draw_handicap")]
    ping_qualified = [r for r in match_results if r["hf_combos"].get("ping_type") in ["严格平系", "均衡平系"] and r["hf_combos"]["首选"]]
    
    report += "### （1）稳健串（低风险 · 推荐主力）\n\n"
    report += "> 选场规则：单场A/B档比赛，排除赔率<1.40的低赔率场，优先跨联赛搭配；总赔率区间：2.4~3.2\n\n"
    if top_parlays:
        for idx, pc in enumerate(top_parlays[:2], 1):
            r1 = pc["r1"]
            r2 = pc["r2"]
            report += f"#### 组合{idx}（{'首选' if idx == 1 else '备选'}）\n"
            report += f"- **对阵1**：[{r1['league_name']}] {r1['home_zh']} vs {r1['away_zh']}（{r1['grade_info']['direction']}，{r1['grade_info']['grade']}，赔率{r1['grade_info']['rec_odds']:.2f}）\n"
            report += f"- **对阵2**：[{r2['league_name']}] {r2['home_zh']} vs {r2['away_zh']}（{r2['grade_info']['direction']}，{r2['grade_info']['grade']}，赔率{r2['grade_info']['rec_odds']:.2f}）\n"
            report += f"- **总赔率**：{pc['total_odds']:.2f}\n"
            report += f"- **综合命中概率**：约{pc['total_prob']:.0%}\n"
            report += "- **组合逻辑**：双稳健档位搭配，跨联赛降低同时爆冷风险，赔率处于合理区间。\n"
            report += "- **建议仓位**：占串关总资金的60%-70%\n\n"
    else:
        report += "> 📌 说明：今日无符合赔率要求（单场≥1.40）的稳健串关组合。\n"
        report += "> 💡 替代方案：优先单场A档重仓，或等待次日更多赛事。\n\n"
    
    report += "### （2）让球价值串（中风险 · 轻仓搭配）\n\n"
    report += "> 选场规则：让球稳胆打底 + 高赔选项搭配；总赔率区间：3.2~4.5\n\n"
    if len(steady_handicap) >= 1 and len(ab_grade) >= 2:
        c1 = steady_handicap[0]
        ab_sorted_by_odds = sorted([r for r in ab_grade if r["grade_info"]["rec_odds"] >= 1.5], key=lambda x: x["grade_info"]["rec_odds"], reverse=True)
        if ab_sorted_by_odds:
            c3 = ab_sorted_by_odds[0]
            total_odds = 1.85 * c3["grade_info"]["rec_odds"]
            report += "#### 组合一\n"
            report += f"- **对阵1**：[{c1['league_name']}] {c1['home_zh']} vs {c1['away_zh']}（{c1['handicap']['favorite_name']}让{c1['handicap']['handicap_goals']:g}球稳胆）\n"
            report += f"- **对阵2**：[{c3['league_name']}] {c3['home_zh']} vs {c3['away_zh']}（收益增强，赔率{c3['grade_info']['rec_odds']:.2f}）\n"
            report += f"- **总赔率**：约{total_odds:.2f}\n"
            report += "- **组合逻辑**：让球稳胆打底+高收益选项搭配，平衡风险与收益。\n"
            report += "- **建议仓位**：占串关总资金的20%-30%\n\n"
        else:
            report += "> 📌 说明：无合适高赔搭配场次。\n\n"
    else:
        report += f"> 📌 说明：本期让球稳胆仅 {len(steady_handicap)} 场，搭配场次不足。\n"
        report += "> 💡 替代方案：单场让球稳胆可单独小注，或搭配高赔率娱乐选项。\n\n"
    
    report += "### （3）半全场串关（中高风险 · 平系专属）\n\n"
    report += "> 选场规则：符合平系标准的比赛（严格+均衡），优先跨联赛搭配；总SP区间：8~15倍\n\n"
    if len(ping_qualified) >= 2:
        best_parlay = build_hf_parlay(ping_qualified, "首选")
        if best_parlay:
            r1 = best_parlay["r1"]
            r2 = best_parlay["r2"]
            c1 = best_parlay["c1"]
            c2 = best_parlay["c2"]
            report += "#### 平系稳健组合（首选）\n"
            report += f"- **对阵1**：[{r1['league_name']}] {r1['home_zh']} vs {r1['away_zh']}（{c1['combo']}）\n"
            report += f"- **对阵2**：[{r2['league_name']}] {r2['home_zh']} vs {r2['away_zh']}（{c2['combo']}）\n"
            report += f"- **总SP赔率**：约 {best_parlay['total_sp']:.1f}\n"
            report += f"- **综合命中概率**：约 {best_parlay['total_prob']*100:.1f}%\n"
            report += "- **组合逻辑**："
            logic_parts = []
            if best_parlay["cross_league"]:
                logic_parts.append("跨联赛搭配，降低同轮相关性")
            logic_parts.append("双平系标准场次，双选容错率叠加")
            logic_parts.append("SP处于黄金收益区间")
            report += "、".join(logic_parts) + "。\n"
            report += "- **建议仓位**：占串关总资金的10%-20%\n\n"
    else:
        report += f"> 📌 说明：本期符合平系标准比赛仅 {len(ping_qualified)} 场，组不成双串。\n"
        report += "> 💡 替代方案：单场平系双选即可，平系策略本身容错率已较高。\n\n"
    
    report += "### （4）娱乐高赔串（高风险 · 纯娱乐）\n\n"
    report += "> ⚠️ 高风险提示：总赔率5.0以上，命中概率低，仅供娱乐。建议仓位不超过总资金1%~2%。\n\n"
    if len(ping_qualified) >= 2:
        gamble_parlay = build_hf_parlay(ping_qualified, "次选")
        if gamble_parlay:
            r1 = gamble_parlay["r1"]
            r2 = gamble_parlay["r2"]
            c1 = gamble_parlay["c1"]
            c2 = gamble_parlay["c2"]
            report += "#### 半全场高赔组合\n"
            report += f"- 对阵1：[{r1['league_name']}] {r1['home_zh']} vs {r1['away_zh']}（{c1['combo']}）\n"
            report += f"- 对阵2：[{r2['league_name']}] {r2['home_zh']} vs {r2['away_zh']}（{c2['combo']}）\n"
            report += f"- 总SP赔率：约 {gamble_parlay['total_sp']:.1f}\n"
            report += "- 说明：次选组合叠加，赔率更高，适合娱乐小注。\n\n"
    else:
        report += "> 📌 说明：平系场次不足，无高赔串关组合。\n\n"
    report += "---\n\n"
    
    # 本期数据规律
    report += "## 📊 本期数据规律（自动统计）\n\n"
    prob_60 = sum(1 for r in match_results if r["grade_info"]["max_prob"] >= 0.6)
    prob_40_50 = sum(1 for r in match_results if 0.4 <= r["grade_info"]["max_prob"] < 0.5)
    prob_35_below = sum(1 for r in match_results if r["grade_info"]["max_prob"] < 0.35)
    report += "### 1. 概率分布\n"
    report += f"- ≥60%稳胆场次：{prob_60} 场\n"
    report += f"- 40%~50%常规场次：{prob_40_50} 场\n"
    report += f"- 35%以下低概率场次：{prob_35_below} 场\n\n"
    grade_a = sum(1 for r in match_results if r["grade_info"]["grade"] == "A档")
    grade_b = sum(1 for r in match_results if r["grade_info"]["grade"] == "B档")
    ev_list = [r["grade_info"]["ev"] for r in match_results if r["grade_info"]["ev"] is not None]
    avg_ev = np.mean(ev_list) if ev_list else 0
    report += "### 2. 价值分布\n"
    report += f"- A档推荐场次：{grade_a} 场\n"
    report += f"- B档推荐场次：{grade_b} 场\n"
    report += f"- 推荐选项平均EV：{avg_ev*100:+.1f}%\n\n"
    report += "### 3. 联赛特征\n"
    league_stats = {}
    for r in match_results:
        ln = r["league_name"]
        if ln not in league_stats:
            league_stats[ln] = {"draw_probs": [], "total_goals": []}
        league_stats[ln]["draw_probs"].append(r["fused_probs"][1])
        league_stats[ln]["total_goals"].append(r["lh"] + r["la"])
    for ln, stats in league_stats.items():
        avg_draw = np.mean(stats["draw_probs"])
        avg_ttg = np.mean(stats["total_goals"])
        report += f"- **{ln}**：平均平局概率 {avg_draw:.1%}，平均总进球 {avg_ttg:.2f}\n"
    report += "\n"
    report += "### 4. 串关资源\n"
    report += f"- B档以上可串比赛：{len(ab_grade)} 场（排除低赔率后{len(ab_grade_for_parlay)}场）\n"
    report += f"- 让球稳胆场次：{len(steady_handicap)} 场（强稳胆{strong_steady_count}场）\n"
    report += f"- 平系策略场次：严格{ping_strict_count}场 + 均衡{ping_balanced_count}场 = {ping_strict_count+ping_balanced_count}场\n"
    report += f"- 战意修正场次：{motivation_adjusted} 场（赛季过滤后）\n\n"
    report += "---\n\n"
    
    # 术语对照表
    report += "## 📚 附录：术语大白话对照表\n\n"
    report += "| 术语 | 大白话解释 |\n|------|------------|\n"
    report += "| 融合概率 | 多个模型综合算出的结果发生概率，数值越高越容易中 |\n"
    report += "| 战意系数 | 根据球队当前赛季排名/分差判断拼命程度，保级/争冠队概率小幅上调，无欲无求队小幅下调 |\n"
    report += "| 多档让球 | 自动计算让0.5/1/1.5/2球四档，选赢盘概率最接近50%的档位，平手盘场次明确说明 |\n"
    report += "| 分级平系 | 半场平局概率≥48%为严格平系（高置信），44%~48%为均衡平系（中等置信），<44%非平系按全场方向推荐 |\n"
    report += "| EV（期望收益） | 长期反复买这个选项，平均每100块能赚多少钱；正数=长期赚，负数=长期亏。多数为负属正常 |\n"
    report += "| SP赔率 | 中了之后的赔付倍数，比如SP=4.0就是投100中了拿400 |\n"
    report += "| 让球盘 | 强队让弱队若干球之后再算胜平负，用来平衡强弱差距 |\n"
    report += "| 平系策略 | 专门挑上半场容易打平的比赛，双选「平平+平X」，胜率更稳 |\n"
    report += "| 二串一 | 两场比赛都中才算赢，赔率是两场相乘，收益更高、难度更大 |\n"
    report += "| 稳胆 | 概率极高的选项，串关里用来打底，降低整体风险 |\n"
    report += "| 市场过热 | 热门方向买的人太多，赔率被压低，长期买性价比低 |\n\n"
    
    print(f"\n✅ 成功生成 {len(match_results)} 场比赛预测")
    if match_results:
        records = []
        for r in match_results:
            record = {
                "match_id": r["match_id"], "date": r["match_time"][:10],
                "league": r["league"], "home_team": r["home_zh"], "away_team": r["away_zh"],
                "fused_home": r["fused_probs"][0], "fused_draw": r["fused_probs"][1], "fused_away": r["fused_probs"][2],
                "pred_direction": r["grade_info"]["direction"], "grade": r["grade_info"]["grade"],
                "ev": r["grade_info"]["ev"], "ht_draw_prob": r["hf_combos"]["半场平概率"],
                "ping_type": r["hf_combos"].get("ping_type", ""),
                "handicap_goals": r["handicap"].get("handicap_goals"),
                "is_draw_handicap": r["handicap"].get("is_draw_handicap", False),
                "handicap_win_prob": r["handicap"]["win_prob"],
                "is_steady": r["handicap"]["is_steady"],
                "home_motivation": r["home_motivation"], "away_motivation": r["away_motivation"],
                "status": "pending"
            }
            records.append(record)
        new_pred_df = pd.DataFrame(records)
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
    print(f"📄 全面修复版报告已生成")

if __name__ == "__main__":
    generate_report()

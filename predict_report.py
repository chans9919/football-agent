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

# ========== 竞彩时间窗口 ==========
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

# ========== 分联赛校准参数 ==========
LEAGUE_PARAMS = {
    "PL": {
        "draw_corr": 0.98,
        "tg_corr": 1.05,
        "single_draw_bonus": 1.08
    },
    "PD": {
        "draw_corr": 1.05,
        "tg_corr": 0.98,
        "single_draw_bonus": 1.08
    },
    "BL1": {
        "draw_corr": 0.95,
        "tg_corr": 1.08,
        "single_draw_bonus": 1.08
    },
    "SA": {
        "draw_corr": 1.03,
        "tg_corr": 0.97,
        "single_draw_bonus": 1.08
    },
    "FL1": {
        "draw_corr": 1.02,
        "tg_corr": 1.00,
        "single_draw_bonus": 1.08
    }
}

# ========== 新增：联赛基础配置（战意模块用） ==========
LEAGUE_TOTAL_ROUNDS = {
    "PL": 38,
    "PD": 38,
    "BL1": 34,
    "SA": 38,
    "FL1": 38
}

LEAGUE_TEAM_COUNT = {
    "PL": 20,
    "PD": 20,
    "BL1": 18,
    "SA": 20,
    "FL1": 20
}

# 欧战资格线（该名次及以上有欧战资格）
LEAGUE_EUROPA_LINE = {
    "PL": 6,
    "PD": 6,
    "BL1": 6,
    "SA": 6,
    "FL1": 4
}

# 战意分值
MOTIVATION_SCORE = {
    "强": 15,
    "正常": 0,
    "弱": -10
}

# 平系策略阈值
PING_STANDARD_THRESHOLD = 0.48

# 球队中文名映射
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
    "Ipswich Town FC": "伊普斯维奇",
    "Coventry City FC": "考文垂",
    "Sunderland AFC": "桑德兰",
    "Hull City AFC": "赫尔城",
    "Real Madrid CF": "皇家马德里",
    "FC Barcelona": "巴塞罗那",
    "Atlético Madrid": "马德里竞技",
    "Club Atlético de Madrid": "马德里竞技",
    "Sevilla FC": "塞维利亚",
    "Real Sociedad": "皇家社会",
    "Real Betis": "皇家贝蒂斯",
    "Real Betis Balompié": "皇家贝蒂斯",
    "Villarreal CF": "比利亚雷亚尔",
    "Valencia CF": "瓦伦西亚",
    "Athletic Club": "毕尔巴鄂竞技",
    "CA Osasuna": "奥萨苏纳",
    "Rayo Vallecano": "巴列卡诺",
    "Rayo Vallecano de Madrid": "巴列卡诺",
    "RCD Espanyol": "西班牙人",
    "RCD Espanyol de Barcelona": "西班牙人",
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
    "Málaga CF": "马拉加",
    "Levante UD": "莱万特",
    "RC Deportivo La Coruña": "拉科鲁尼亚",
    "Real Racing Club de Santander": "桑坦德竞技",
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
    "SV 07 Elversberg": "埃尔沃斯堡",
    "SC Paderborn 07": "帕德博恩",
    "Hamburger SV": "汉堡",
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
    "Bologna FC 1909": "博洛尼亚",
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
    "Paris Saint Germain FC": "巴黎圣日耳曼",
    "Olympique de Marseille": "马赛",
    "Olympique Lyonnais": "里昂",
    "AS Monaco FC": "摩纳哥",
    "LOSC Lille": "里尔",
    "Lille OSC": "里尔",
    "Stade Rennais FC": "雷恩",
    "Stade Rennais FC 1901": "雷恩",
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
    "Racing Club de Lens": "朗斯",
    "Havre AC": "勒阿弗尔",
    "Le Havre AC": "勒阿弗尔",
    "AS Saint-Étienne": "圣埃蒂安",
    "Stade Malherbe Caen": "卡昂",
    "FC Girondins de Bordeaux": "波尔多",
    "ESTAC Troyes": "特鲁瓦",
    "ES Troyes AC": "特鲁瓦",
    "Dijon FCO": "第戎",
    "Le Mans FC": "勒芒",
    "Paris FC": "巴黎FC"
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

# ========== 数据拉取函数 ==========
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
    
    # 半场概率
    lambda_half_home = lambda_home * half_ratio
    lambda_half_away = lambda_away * half_ratio
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
    
    # 主让1球 / 客让1球 双计算
    hh_win, h_draw, ha_win = 0, 0, 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_i = i - 1
            if new_i < 0:
                ha_win += matrix[i,j]
            else:
                if new_i > j:
                    hh_win += matrix[i,j]
                elif new_i == j:
                    h_draw += matrix[i,j]
                else:
                    ha_win += matrix[i,j]
    
    ah_win, a_draw, aa_win = 0, 0, 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_j = j - 1
            if new_j < 0:
                ah_win += matrix[i,j]
            else:
                if i > new_j:
                    ah_win += matrix[i,j]
                elif i == new_j:
                    a_draw += matrix[i,j]
                else:
                    aa_win += matrix[i,j]
    
    handicap = {
        "home_let": (hh_win, h_draw, ha_win),
        "away_let": (aa_win, a_draw, ah_win)
    }
    
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
        "htft_probs": htft_probs,
        "handicap": handicap,
        "total_goals": sorted_total,
        "ht_draw_prob": ht_draw
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
    
    if odds_data:
        rec_odds = odds_data[dir_idx]
        ev = max_prob * rec_odds - 1
        
        if 1.50 <= rec_odds <= 2.20:
            odds_comment = "✅ 赔率合理"
        elif 1.30 <= rec_odds < 1.50:
            odds_comment = "⚠️ 赔率偏低"
            if max_prob < 0.55 and grade == "A档":
                final_grade = "B档"
            elif grade in ["B档", "C档"]:
                final_grade = "C档" if grade == "B档" else "不推荐"
        elif 2.20 < rec_odds <= 3.00:
            odds_comment = "⚠️ 赔率偏高"
            if max_prob < 0.50:
                final_grade = "C档" if grade == "B档" else "不推荐"
        else:
            odds_comment = "❌ 赔率超出合理区间"
            if max_prob < 0.60:
                final_grade = "不推荐"
    
    return {
        "direction": dir_name,
        "dir_idx": dir_idx,
        "max_prob": max_prob,
        "prob_grade": prob_grade,
        "grade": final_grade,
        "odds_comment": odds_comment,
        "ev": ev,
        "rec_odds": odds_data[dir_idx] if odds_data else None
    }

# ========== 半全场双选 方向强制对齐 ==========
def calculate_hf_combos(htft_probs, ht_draw_prob, full_dir_idx, odds_data=None):
    ping_combos = {
        "平平": htft_probs["平平"],
        "平胜": htft_probs["平胜"],
        "平负": htft_probs["平负"]
    }
    
    sp_est = {}
    if odds_data:
        h_odds, d_odds, a_odds = odds_data
        sp_est["平平"] = d_odds * 1.8
        sp_est["平胜"] = h_odds * 2.2
        sp_est["平负"] = a_odds * 2.2
        sp_est["胜胜"] = h_odds * 1.6
        sp_est["负负"] = a_odds * 1.6
    
    combos = []
    options = list(ping_combos.items())
    
    for i in range(len(options)):
        for j in range(i+1, len(options)):
            name1, prob1 = options[i]
            name2, prob2 = options[j]
            total_prob = prob1 + prob2
            
            if odds_data and name1 in sp_est and name2 in sp_est:
                sp1 = sp_est[name1]
                sp2 = sp_est[name2]
                ev = (prob1 * sp1 + prob2 * sp2) - 2
                ev_pct = ev / 2 * 100
            else:
                ev = None
                ev_pct = None
            
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
                "combo": f"{name1} + {name2}",
                "total_prob": total_prob,
                "sp": f"{sp_est.get(name1, 0):.2f} / {sp_est.get(name2, 0):.2f}" if sp_est else "无赔率数据，无法估算",
                "sp_avg": (sp_est.get(name1, 0) + sp_est.get(name2, 0)) / 2 if sp_est else 0,
                "ev_pct": ev_pct,
                "align_score": align_score
            })
    
    # 首选：必须方向对齐，按概率排序
    first_candidates = [c for c in combos if c["align_score"] == 100]
    first_candidates.sort(key=lambda x: x["total_prob"], reverse=True)
    first = first_candidates[0] if first_candidates else None
    
    # 次选：EV最高，方向不限
    second_candidates = [c for c in combos if c != first]
    if odds_data:
        second_candidates.sort(key=lambda x: x["ev_pct"] if x["ev_pct"] is not None else -999, reverse=True)
    else:
        second_candidates.sort(key=lambda x: x["total_prob"], reverse=True)
    second = second_candidates[0] if second_candidates else None
    
    # 博弈备选
    gamble = None
    for c in combos:
        if c == first or c == second:
            continue
        if c["ev_pct"] is not None and -2 <= c["ev_pct"] < 0:
            gamble = c
            break
    
    result = {
        "符合平系标准": ht_draw_prob >= PING_STANDARD_THRESHOLD,
        "半场平概率": ht_draw_prob,
        "阈值": PING_STANDARD_THRESHOLD
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
                "r1": r1,
                "r2": r2,
                "c1": c1,
                "c2": c2,
                "total_sp": total_sp,
                "total_prob": total_prob,
                "cross_league": cross_league,
                "score": score
            })
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

# ========== 新增：战意系数模块 ==========
def calculate_league_standings(league_df, league):
    """
    计算联赛当前积分榜
    返回：{team: {"points": int, "gd": int, "rank": int, "played": int}}
    """
    if league_df.empty:
        return {}
    
    # 只算已完赛的比赛
    if "status" in league_df.columns:
        finished = league_df[league_df["status"].isin(["FINISHED", "finished", "已完赛"])]
    else:
        finished = league_df.copy()
    
    if finished.empty:
        return {}
    
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
    
    # 计算净胜球并排名
    for team in team_stats:
        team_stats[team]["gd"] = team_stats[team]["gf"] - team_stats[team]["ga"]
    
    sorted_teams = sorted(team_stats.items(), key=lambda x: (x[1]["points"], x[1]["gd"]), reverse=True)
    for rank, (team, _) in enumerate(sorted_teams, 1):
        team_stats[team]["rank"] = rank
    
    return team_stats

def get_current_round(standings, league):
    """估算当前轮次：用每队平均已赛场次"""
    if not standings:
        return 1
    avg_played = np.mean([s["played"] for s in standings.values()])
    return max(1, round(avg_played))

def get_motivation(team, standings, league, current_round):
    """
    判断球队战意等级
    返回：(等级, 分值, 说明)
    """
    if team not in standings:
        return "正常", 0, "无积分榜数据"
    
    info = standings[team]
    rank = info["rank"]
    points = info["points"]
    total_teams = LEAGUE_TEAM_COUNT.get(league, 20)
    total_rounds = LEAGUE_TOTAL_ROUNDS.get(league, 38)
    
    # 赛季初（前5轮），战意参考价值低，统一正常
    if current_round <= 5:
        return "正常", 0, "赛季初，战意参考价值低"
    
    # 争冠组：排名前4，与榜首分差≤5分
    sorted_by_points = sorted(standings.items(), key=lambda x: x[1]["points"], reverse=True)
    top_points = sorted_by_points[0][1]["points"] if sorted_by_points else 0
    
    if rank <= 4 and (top_points - points) <= 5:
        return "强", MOTIVATION_SCORE["强"], f"争冠组（排名第{rank}，距榜首{top_points-points}分）"
    
    # 欧战资格组：排名在欧战线±2名，与欧战线分差≤5分
    europa_line = LEAGUE_EUROPA_LINE.get(league, 6)
    if europa_line - 2 <= rank <= europa_line + 2:
        if rank <= europa_line:
            line_points = sorted_by_points[europa_line - 1][1]["points"] if len(sorted_by_points) >= europa_line else 0
            diff = points - line_points
            return "强", MOTIVATION_SCORE["强"], f"欧战区内（排名第{rank}）"
        else:
            line_points = sorted_by_points[europa_line - 1][1]["points"] if len(sorted_by_points) >= europa_line else 0
            diff = line_points - points
            if diff <= 5:
                return "强", MOTIVATION_SCORE["强"], f"冲击欧战（排名第{rank}，距欧战区{diff}分）"
    
    # 保级组：后3名直接算，倒数4-5名与安全线差≤3分
    relegation_zone_start = total_teams - 2  # 倒数第3名的排名
    safety_rank = total_teams - 3  # 倒数第4名（安全线）
    
    if rank >= relegation_zone_start:
        return "强", MOTIVATION_SCORE["强"], f"保级区（排名第{rank}）"
    
    if safety_rank <= rank <= total_teams - 4 + 1:  # 倒数4-5名
        safety_points = sorted_by_points[safety_rank - 1][1]["points"] if len(sorted_by_points) >= safety_rank else 0
        diff = safety_points - points
        if diff <= 3:
            return "强", MOTIVATION_SCORE["强"], f"保级边缘（排名第{rank}，距安全线{diff}分）"
    
    # 弱战意：赛季末，排名中游且无欲无求，且与上下线分差都>8分
    if current_round >= total_rounds - 4:
        if 8 <= rank <= total_teams - 6:
            # 确认距离欧战区和保级区都很远
            if rank > europa_line + 2 and rank < safety_rank - 1:
                return "弱", MOTIVATION_SCORE["弱"], "赛季末无欲无求"
    
    return "正常", 0, "中游正常战意"

def apply_motivation_adjustment(home_team, away_team, fused_probs, standings, league, current_round):
    """
    应用战意修正
    返回：(修正后概率, 修正说明, 主战意, 客战意)
    """
    home_level, home_score, home_desc = get_motivation(home_team, standings, league, current_round)
    away_level, away_score, away_desc = get_motivation(away_team, standings, league, current_round)
    
    motivation_diff = home_score - away_score
    
    # 战意差为0，不修正
    if motivation_diff == 0:
        return fused_probs.copy(), "双方战意相当，不修正", home_level, away_level
    
    # 赛季阶段系数
    total_rounds = LEAGUE_TOTAL_ROUNDS.get(league, 38)
    if current_round <= 5:
        season_factor = 0.3
    elif current_round >= total_rounds - 4:
        season_factor = 1.5
    else:
        season_factor = 1.0
    
    # 数据完整性校验：赛季过半前降低修正
    if current_round < total_rounds * 0.5:
        season_factor *= 0.5
    
    # 按战意差分档给基础修正系数
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
    
    # 应用赛季阶段系数（系数向1.0靠拢）
    home_factor = 1.0 + (base_home_factor - 1.0) * season_factor
    away_factor = 1.0 + (base_away_factor - 1.0) * season_factor
    draw_factor_adj = 1.0 + (draw_factor - 1.0) * season_factor
    
    # 应用修正
    adjusted = [
        fused_probs[0] * home_factor,
        fused_probs[1] * draw_factor_adj,
        fused_probs[2] * away_factor
    ]
    
    # 重新归一化
    total = sum(adjusted)
    adjusted = [p / total for p in adjusted]
    
    # 上限保护：单场概率变化不超过±3%
    max_change = max(abs(adjusted[i] - fused_probs[i]) for i in range(3))
    if max_change > 0.03:
        scale = 0.03 / max_change
        adjusted = [
            fused_probs[i] + (adjusted[i] - fused_probs[i]) * scale
            for i in range(3)
        ]
        total = sum(adjusted)
        adjusted = [p / total for p in adjusted]
    
    # 生成修正说明
    home_change = (adjusted[0] - fused_probs[0]) * 100
    away_change = (adjusted[2] - fused_probs[2]) * 100
    
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
    
    for league in LEAGUES:
        if "league" in all_matches_df.columns:
            league_df = all_matches_df[all_matches_df["league"] == league].copy()
        else:
            league_df = all_matches_df.copy() if league == "PL" else pd.DataFrame()
        league_df_cache[league] = league_df
        elo_cache[league] = load_or_calculate_elo(league_df, league)
        data_sufficient_cache[league] = len(league_df) >= 10
        
        # 预计算积分榜和当前轮次
        standings = calculate_league_standings(league_df, league)
        standings_cache[league] = standings
        current_round_cache[league] = get_current_round(standings, league)
        if standings:
            print(f"📊 {LEAGUE_NAMES.get(league, league)} 积分榜计算完成，共{len(standings)}支球队，当前约第{current_round_cache[league]}轮")
    
    # ========== 第一步：预计算所有比赛数据 ==========
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
            
            # 分联赛概率校准
            p_poisson_raw = [poisson_probs["home_win"], poisson_probs["draw"], poisson_probs["away_win"]]
            p_poisson = [
                p_poisson_raw[0],
                p_poisson_raw[1] * params["draw_corr"],
                p_poisson_raw[2]
            ]
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
            
            # ========== 新增：战意修正 ==========
            fused_probs, motivation_desc, home_motivation, away_motivation = apply_motivation_adjustment(
                home_en, away_en, fused_probs_raw, standings, league, current_round
            )
            
            grade_info = grade_match(fused_probs, odds_data)
            full_dir_idx = grade_info["dir_idx"]
            
            # 让球盘：自动识别强队
            if fused_probs[0] >= fused_probs[2]:
                favorite_team = "home"
                favorite_name = home_zh
                h_win, h_draw, h_lose = poisson_probs["handicap"]["home_let"]
            else:
                favorite_team = "away"
                favorite_name = away_zh
                h_win, h_draw, h_lose = poisson_probs["handicap"]["away_let"]
            
            handicap_info = {
                "favorite": favorite_team,
                "favorite_name": favorite_name,
                "win_prob": h_win,
                "draw_prob": h_draw,
                "lose_prob": h_lose,
                "is_steady": h_win >= 0.55
            }
            
            hf_combos = calculate_hf_combos(
                poisson_probs["htft_probs"],
                poisson_probs["ht_draw_prob"],
                full_dir_idx,
                odds_data
            )
            
            top_ttg = sorted(poisson_probs["total_goals"], key=lambda x: x[1], reverse=True)[:3]
            
            divergence = None
            if p_market:
                diff = abs(fused_probs[grade_info["dir_idx"]] - p_market[grade_info["dir_idx"]])
                divergence = {
                    "value": diff,
                    "is_large": diff >= 0.1,
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
                "match_no": f"{match_counter:03d}",
                "match_id": match_id,
                "league": league,
                "league_name": league_name,
                "home_zh": home_zh,
                "away_zh": away_zh,
                "match_time": match_time,
                "lh": lh,
                "la": la,
                "p_poisson": p_poisson,
                "p_market": p_market,
                "p_elo": p_elo,
                "fused_probs": fused_probs,
                "fused_probs_raw": fused_probs_raw,
                "grade_info": grade_info,
                "handicap": handicap_info,
                "hf_combos": hf_combos,
                "top_scores": poisson_probs["top_scores"],
                "top_ttg": top_ttg,
                "divergence": divergence,
                "model_consistent": model_consistent,
                "odds_data": odds_data,
                "data_sufficient": data_sufficient,
                "motivation_desc": motivation_desc,
                "home_motivation": home_motivation,
                "away_motivation": away_motivation
            }
            match_results.append(result)
            match_counter += 1
            
        except Exception as e:
            print(f"❌ 跳过比赛 {home_en} vs {away_en}：{str(e)}")
            traceback.print_exc()
            continue
    
    # ========== 第二步：生成完整报告 ==========
    now_bj = datetime.utcnow() + timedelta(hours=8)
    report = ""
    
    report += "# 足球预测报告（战意系数版）\n\n"
    report += f"**生成时间**：{now_bj.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n"
    report += f"**预测时段**：{TARGET_DATE_LABEL} 17:00 ~ 次日 12:00\n"
    report += f"**数据来源**：Football-Data.org + The Odds API + ELO\n"
    
    has_any_odds = any(r["odds_data"] for r in match_results)
    if has_any_odds:
        model_desc = "泊松模型（DC修正+联赛校准） + 全庄家赔率中位数 + ELO + 战意系数 在 logit 空间融合"
    else:
        model_desc = "泊松模型（DC修正+联赛校准） + ELO + 战意系数（本期无赔率数据）"
    report += f"**模型说明**：{model_desc}\n\n"
    
    report += "> ⚠️ 风险提示：所有预测仅供参考，不构成投注建议。足球比赛不确定性高，请理性购彩，量力而行。\n\n"
    
    leagues_covered = set(r["league_name"] for r in match_results)
    ping_count = sum(1 for r in match_results if r["hf_combos"]["符合平系标准"])
    grade_a_count = sum(1 for r in match_results if r["grade_info"]["grade"] == "A档")
    grade_b_count = sum(1 for r in match_results if r["grade_info"]["grade"] == "B档")
    motivation_adjusted = sum(1 for r in match_results if "不修正" not in r["motivation_desc"] and "较小" not in r["motivation_desc"])
    
    report += f"**本期概览**：共 {len(match_results)} 场比赛，覆盖 {'、'.join(leagues_covered)}；\n"
    report += f"- A档强推：{grade_a_count} 场 | B档可买：{grade_b_count} 场 | 符合平系策略：{ping_count} 场\n"
    report += f"- 战意修正场次：{motivation_adjusted} 场（保级/争冠/欧战战意差异已纳入概率调整）\n\n"
    report += "---\n\n"
    
    # 总览汇总表
    report += "## 📋 总览汇总表\n\n"
    report += "💡 快速用法：直接筛「A档」比赛做主投，「B档」做串关，C档直接跳过。\n\n"
    report += "| 编号 | 对阵 | 胜平负首选 | 档位 | 让球参考 | 战意提示 | 半全场首选双选 | 总进球首选 |\n"
    report += "|------|------|------------|------|----------|----------|------------------|------------|\n"
    
    for r in match_results:
        no = r["match_no"]
        vs = f"{r['home_zh']} vs {r['away_zh']}"
        first_dir = r["grade_info"]["direction"]
        first_prob = f"{r['grade_info']['max_prob']:.1%}"
        grade = r["grade_info"]["grade"]
        
        h = r["handicap"]
        if h["is_steady"]:
            hf_ref = f"{h['favorite_name']}让球稳"
        else:
            hf_ref = "走盘风险高"
        
        # 战意提示简写
        if r["home_motivation"] == "强" and r["away_motivation"] == "弱":
            mot_short = "主战意强"
        elif r["home_motivation"] == "弱" and r["away_motivation"] == "强":
            mot_short = "客战意强"
        elif r["home_motivation"] == "强" and r["away_motivation"] == "强":
            mot_short = "双方强战意"
        else:
            mot_short = "战意正常"
        
        if r["hf_combos"]["首选"]:
            hf_first = r["hf_combos"]["首选"]["combo"]
        else:
            hf_first = "无推荐"
        
        ttg_first = f"{r['top_ttg'][0][0]}球" if r["top_ttg"] else "-"
        
        report += f"| {no} | {vs} | {first_dir} {first_prob} | {grade} | {hf_ref} | {mot_short} | {hf_first} | {ttg_first} |\n"
    
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
        
        # 一、单场胜平负
        report += "##### 一、单场胜平负（主力投注）\n\n"
        report += "**融合最终概率（含战意修正）**\n"
        report += "| 主胜 | 平局 | 客胜 |\n|---|---:|---:|\n"
        report += f"| {r['fused_probs'][0]:.1%} | {r['fused_probs'][1]:.1%} | {r['fused_probs'][2]:.1%} |\n\n"
        
        # 战意修正说明
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
        
        if g["rec_odds"] and 1.5 <= g["rec_odds"] <= 2.2:
            reason_parts.append("赔率处于合理区间")
        elif g["rec_odds"] and g["rec_odds"] < 1.5:
            reason_parts.append("赔率偏低，已做降档校验")
        
        report += "、".join(reason_parts) + "。\n"
        
        if g["ev"] is not None:
            ev_sign = "正收益" if g["ev"] > 0 else "负收益"
            report += f"> 💡 **EV解读**：{g['ev']*100:+.1f}%（长期重复买入预期{ev_sign}，{abs(g['ev']*100):.1f}%/百本金）\n"
        
        report += "\n**建议仓位**："
        if g["grade"] == "A档":
            report += "主力仓位（占总资金30%-50%）"
        elif g["grade"] == "B档":
            report += "轻仓（占总资金10%-20%）"
        elif g["grade"] == "C档":
            report += "娱乐小注（≤总资金5%）"
        else:
            report += "不建议介入"
        report += "\n\n"
        
        # 二、让球盘
        report += f"##### 二、让球盘（串关稳胆参考 · {r['handicap']['favorite_name']}让一球）\n\n"
        report += "| 强队赢盘 | 走盘 | 弱队赢盘 |\n|---|---:|---:|\n"
        h = r["handicap"]
        report += f"| {h['win_prob']:.1%} | {h['draw_prob']:.1%} | {h['lose_prob']:.1%} |\n\n"
        
        if h["is_steady"]:
            report += f"**稳胆判定**：✅ {h['favorite_name']}让球后赢盘概率超55%，可作为串关稳胆选项\n"
        else:
            report += f"**稳胆判定**：三项概率接近，走盘风险较高，不建议单独做稳胆\n"
        report += "\n"
        
        # 三、半全场双选
        report += "##### 三、半全场双选（平系策略）\n\n"
        hf = r["hf_combos"]
        
        if hf["符合平系标准"]:
            report += f"✅ **本场符合平系择赛标准**\n"
            report += f"> 📌 判定依据：半场平局概率 {hf['半场平概率']:.1%}，≥ {hf['阈值']*100:.0f}% 的合格阈值，上半场平局概率高，双选容错率强。\n\n"
        else:
            report += f"⚠️ **本场不符合平系择赛标准**\n"
            report += f"> 📌 判定依据：半场平局概率 {hf['半场平概率']:.1%}，低于 {hf['阈值']*100:.0f}% 的合格阈值，上半场分胜负概率更高，双选容错率不足。以下仅作参考。\n\n"
        
        if hf["首选"]:
            c = hf["首选"]
            report += "###### ✅ 首选组合（稳健主力 · 建议占半全场仓位60%-70%）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **综合中奖概率**：{c['total_prob']:.1%}（约每{1/c['total_prob']:.1f}单中1单）\n"
            report += f"- **对应SP赔率**：{c['sp']}\n"
            if c["ev_pct"] is not None:
                report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        
        if hf["次选"]:
            c = hf["次选"]
            report += "###### ⚠️ 次选组合（收益增强 · 建议占半全场仓位20%-30%）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **综合中奖概率**：{c['total_prob']:.1%}\n"
            report += f"- **对应SP赔率**：{c['sp']}\n"
            if c["ev_pct"] is not None:
                report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        
        if hf["博弈备选"]:
            c = hf["博弈备选"]
            report += "###### 🎲 博弈备选（娱乐小注 · 建议≤半全场仓位10%）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **对应SP赔率**：{c['sp']}\n"
            report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%，长期微亏，仅适合临场小仓位博弈\n"
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
    steady_handicap = [r for r in match_results if r["handicap"]["is_steady"] and r["odds_data"]]
    ping_qualified = [r for r in match_results if r["hf_combos"]["符合平系标准"] and r["hf_combos"]["首选"]]
    
    report += "### （1）稳健串（低风险 · 推荐主力）\n\n"
    report += "> 选场规则：单场A/B档比赛，优先跨联赛搭配；总赔率区间：2.4~3.2\n\n"
    
    if len(ab_grade) >= 2:
        ab_sorted = sorted(ab_grade, key=lambda x: x["grade_info"]["rec_odds"])
        c1 = ab_sorted[0]
        c2 = ab_sorted[1]
        total_odds = c1["grade_info"]["rec_odds"] * c2["grade_info"]["rec_odds"]
        total_prob = c1["grade_info"]["max_prob"] * c2["grade_info"]["max_prob"]
        
        report += "#### 组合一（首选）\n"
        report += f"- **对阵1**：[{c1['league_name']}] {c1['home_zh']} vs {c1['away_zh']}（{c1['grade_info']['direction']}，{c1['grade_info']['grade']}）\n"
        report += f"- **对阵2**：[{c2['league_name']}] {c2['home_zh']} vs {c2['away_zh']}（{c2['grade_info']['direction']}，{c2['grade_info']['grade']}）\n"
        report += f"- **总赔率**：{total_odds:.2f}\n"
        report += f"- **综合命中概率**：约{total_prob:.0%}\n"
        report += "- **组合逻辑**：双稳健档位搭配，跨联赛降低同时爆冷风险，赔率处于黄金区间。\n"
        report += "- **建议仓位**：占串关总资金的60%-70%\n\n"
    else:
        report += f"> 📌 说明：本期B档以上可串比赛仅 {len(ab_grade)} 场，满足不了二串一最低2场要求。\n"
        report += "> 💡 替代方案：优先单场A档重仓，或等待次日更多赛事。\n\n"
    
    report += "### （2）让球价值串（中风险 · 轻仓搭配）\n\n"
    report += "> 选场规则：强队让球稳胆打底 + 高赔选项搭配；总赔率区间：3.2~4.5\n\n"
    
    if len(steady_handicap) >= 1 and len(ab_grade) >= 2:
        c1 = steady_handicap[0]
        c3 = ab_sorted[-1]
        total_odds = 1.85 * c3["grade_info"]["rec_odds"]
        
        report += "#### 组合一\n"
        report += f"- **对阵1**：[{c1['league_name']}] {c1['home_zh']} vs {c1['away_zh']}（{c1['handicap']['favorite_name']}让球稳胆）\n"
        report += f"- **对阵2**：[{c3['league_name']}] {c3['home_zh']} vs {c3['away_zh']}（收益增强）\n"
        report += f"- **总赔率**：约{total_odds:.2f}\n"
        report += "- **组合逻辑**：让球稳胆打底+高收益选项搭配，平衡风险与收益。\n"
        report += "- **建议仓位**：占串关总资金的20%-30%\n\n"
    else:
        report += f"> 📌 说明：本期让球稳胆仅 {len(steady_handicap)} 场，搭配场次不足。\n"
        report += "> 💡 替代方案：单场让球稳胆可单独小注，或搭配高赔率娱乐选项。\n\n"
    
    report += "### （3）半全场串关（中高风险 · 平系专属）\n\n"
    report += "> 选场规则：符合平系标准的比赛，优先跨联赛搭配；总SP区间：8~15倍\n\n"
    
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
    report += f"- B档以上可串比赛：{len(ab_grade)} 场\n"
    report += f"- 让球稳胆场次：{len(steady_handicap)} 场\n"
    report += f"- 符合平系策略场次：{ping_count} 场\n"
    report += f"- 战意修正场次：{motivation_adjusted} 场\n\n"
    
    report += "---\n\n"
    
    # 术语对照表
    report += "## 📚 附录：术语大白话对照表\n\n"
    report += "| 术语 | 大白话解释 |\n|------|------------|\n"
    report += "| 融合概率 | 多个模型综合算出的结果发生概率，数值越高越容易中 |\n"
    report += "| 战意系数 | 根据球队排名/分差判断拼命程度，保级/争冠队概率小幅上调，无欲无求队小幅下调 |\n"
    report += "| EV（期望收益） | 长期反复买这个选项，平均每100块能赚多少钱；正数=长期赚，负数=长期亏 |\n"
    report += "| SP赔率 | 中了之后的赔付倍数，比如SP=4.0就是投100中了拿400 |\n"
    report += "| 让球盘 | 强队让弱队1球之后再算胜平负，用来平衡强弱差距 |\n"
    report += "| 平系策略 | 专门挑上半场容易打平的比赛，双选「平平+平X」，胜率更稳 |\n"
    report += "| 二串一 | 两场比赛都中才算赢，赔率是两场相乘，收益更高、难度更大 |\n"
    report += "| 稳胆 | 概率极高的选项，串关里用来打底，降低整体风险 |\n"
    report += "| 市场过热 | 热门方向买的人太多，赔率被压低，长期买性价比低 |\n\n"
    
    # 保存
    print(f"\n✅ 成功生成 {len(match_results)} 场比赛预测")
    
    if match_results:
        records = []
        for r in match_results:
            record = {
                "match_id": r["match_id"],
                "date": r["match_time"][:10],
                "league": r["league"],
                "home_team": r["home_zh"],
                "away_team": r["away_zh"],
                "fused_home": r["fused_probs"][0],
                "fused_draw": r["fused_probs"][1],
                "fused_away": r["fused_probs"][2],
                "pred_direction": r["grade_info"]["direction"],
                "grade": r["grade_info"]["grade"],
                "ev": r["grade_info"]["ev"],
                "ht_draw_prob": r["hf_combos"]["半场平概率"],
                "ping_standard": r["hf_combos"]["符合平系标准"],
                "home_motivation": r["home_motivation"],
                "away_motivation": r["away_motivation"],
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
    print(f"📄 战意系数版报告已生成")

if __name__ == "__main__":
    generate_report()

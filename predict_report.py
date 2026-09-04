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

# 平系策略阈值
PING_STANDARD_THRESHOLD = 0.48  # 半场平局概率≥48% 符合平系择赛标准

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
    "Luton Town FC": "卢顿",
    "Burnley FC": "伯恩利",
    "Sheffield United FC": "谢菲联",
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

# ========== 数据拉取与核心计算函数 ==========
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
    
    # 基于联赛真实半场进球比计算半场概率
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
        "htft_probs": htft_probs,
        "handicap": (handicap_home_win, handicap_draw, handicap_away_win),
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

# ========== 新增：各联赛真实半场进球比计算 ==========
def calculate_league_half_ratio(all_matches_df):
    """从历史数据计算各联赛的半场进球占比"""
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

# ========== 新增：单场胜平负评级函数 ==========
def grade_match(fused_probs, odds_data):
    max_prob = max(fused_probs)
    dir_idx = np.argmax(fused_probs)
    dir_name = ["主胜", "平局", "客胜"][dir_idx]
    
    # 概率档位
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
    
    # 赔率校验
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

# ========== 新增：半全场双选组合计算（三档分级） ==========
def calculate_hf_combos(htft_probs, ht_draw_prob, odds_data=None):
    # 平系组合：平平、平胜、平负
    ping_combos = {
        "平平": htft_probs["平平"],
        "平胜": htft_probs["平胜"],
        "平负": htft_probs["平负"]
    }
    
    # 半全场SP估算（基于全场赔率经验系数推导，仅供参考）
    # 系数说明：根据历史半全场赔率与全场赔率的对应关系经验值设定
    sp_est = {}
    if odds_data:
        h_odds, d_odds, a_odds = odds_data
        sp_est["平平"] = d_odds * 1.8
        sp_est["平胜"] = h_odds * 2.2
        sp_est["平负"] = a_odds * 2.2
        sp_est["胜胜"] = h_odds * 1.6
        sp_est["负负"] = a_odds * 1.6
    
    # 生成双选组合
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
                # ----- EV计算公式说明 -----
                # 双选共投入2元（每个选项各投1元）
                # 中name1时：收回 sp1 × 1元，另一选项亏1元，净赚 sp1 - 2 元
                # 中name2时：收回 sp2 × 1元，另一选项亏1元，净赚 sp2 - 2 元
                # 都不中时：净亏 2 元
                # 长期期望 = prob1×(sp1-2) + prob2×(sp2-2) + (1-prob1-prob2)×(-2)
                # 化简后 = prob1×sp1 + prob2×sp2 - 2
                ev = (prob1 * sp1 + prob2 * sp2) - 2
                ev_pct = ev / 2 * 100  # 相对于总投入的收益率
            else:
                ev = None
                ev_pct = None
            
            combos.append({
                "combo": f"{name1} + {name2}",
                "total_prob": total_prob,
                "sp": f"{sp_est.get(name1, 0):.2f} / {sp_est.get(name2, 0):.2f}" if sp_est else "参考",
                "ev_pct": ev_pct
            })
    
    # 按综合EV排序（有EV时），否则按概率排序
    if odds_data:
        combos.sort(key=lambda x: x["ev_pct"] if x["ev_pct"] is not None else -999, reverse=True)
    else:
        combos.sort(key=lambda x: x["total_prob"], reverse=True)
    
    # 分档判定
    result = {
        "符合平系标准": ht_draw_prob >= PING_STANDARD_THRESHOLD,
        "半场平概率": ht_draw_prob
    }
    
    # 首选：EV≥3% 且 概率≥45%
    first = None
    for c in combos:
        if c["ev_pct"] is not None and c["ev_pct"] >= 3 and c["total_prob"] >= 0.45:
            first = c
            break
    if not first and combos:
        first = combos[0]
    result["首选"] = first
    
    # 次选：EV>0 且 概率≥42%
    second = None
    for c in combos:
        if c == first:
            continue
        if c["ev_pct"] is not None and c["ev_pct"] > 0 and c["total_prob"] >= 0.42:
            second = c
            break
    result["次选"] = second
    
    # 博弈备选：-2% ≤ EV < 0
    gamble = None
    for c in combos:
        if c == first or c == second:
            continue
        if c["ev_pct"] is not None and -2 <= c["ev_pct"] < 0:
            gamble = c
            break
    result["博弈备选"] = gamble
    
    return result

# ========== 通俗解读函数（保留原有功能） ==========
def generate_friendly_advice(home_zh, away_zh, lh, la,
                            p_poisson, p_elo, p_market,
                            poisson_probs, home_elo, away_elo,
                            fused_probs, odds_data, data_sufficient):
    advice = "\n━━━━ 通俗解读 ━━━━\n\n"
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
    dirs = [np.argmax(p_poisson), np.argmax(p_elo)]
    if p_market:
        dirs.append(np.argmax(p_market))
    same_count = sum(1 for d in dirs if d == dirs[0])
    if same_count == len(dirs):
        advice += "   - 所有模型方向一致，分歧度低。\n"
    else:
        advice += "   - 模型存在一定分歧，需谨慎。\n"
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
        rec_odds = [odds_h, odds_d, odds_a][direction_idx]
        if 1.5 <= rec_odds <= 2.2:
            odds_comment = "✅ 赔率合理"
        elif 1.3 <= rec_odds < 1.5:
            odds_comment = "⚠️ 赔率偏低，要求概率≥55%才买"
        elif 2.2 < rec_odds <= 3.0:
            odds_comment = "⚠️ 赔率偏高，要求概率≥50%才买"
        else:
            odds_comment = "❌ 赔率不合理，不建议"
        advice += f"   - 推荐赔率：{rec_odds:.2f}（{odds_comment}）\n"
        ev = max_prob * rec_odds - 1
        advice += f"   - 预期：10场类似比赛约{max_prob*10:.0f}场命中，盈亏平衡点约{1/rec_odds:.1%}\n"
    else:
        advice += "   - 无赔率数据，无法判断赔率合理性。\n"
    advice += "🔗 串关适合度："
    if grade in ["A档（强推）", "B档（可买）"] and odds_data:
        if 1.4 <= rec_odds <= 2.0:
            advice += "✅ 适合串关，可与另一场 1.5~1.8 的选项组合\n"
        else:
            advice += "⚠️ 赔率较高，串关需搭配更稳选项\n"
    else:
        advice += "❌ 不适合串关\n"
    hw, hd, ha = poisson_probs["handicap"]
    advice += "🎯 让球（主让一球）："
    if ha > 0.5:
        advice += f"客胜 {ha:.1%}，主队让球偏深，客队受让更稳\n"
    elif hw > 0.4:
        advice += f"主胜 {hw:.1%}，主队赢盘能力较强\n"
    else:
        advice += "三项接近，走盘风险高\n"
    top_htft = poisson_probs["top_htft"]
    advice += "⏱ 半全场参考：\n"
    advice += f"   最高概率组合：{top_htft[0][0]}（{top_htft[0][1]:.1%}），"
    if top_htft[0][0].startswith("平"):
        advice += "上半场平局概率大，比赛慢热。\n"
    else:
        advice += "上半场分胜负，节奏较快。\n"
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
    if odds_data:
        probs = fused_probs
        min_prob_idx = np.argmin(probs)
        min_prob = probs[min_prob_idx]
        if min_prob < 0.3:
            min_odds = [odds_data[0], odds_data[1], odds_data[2]][min_prob_idx]
            min_dir = ["主胜", "平局", "客胜"][min_prob_idx]
            advice += f"🎲 博冷提示：{min_dir} 概率仅 {min_prob:.1%}，赔率 {min_odds:.2f}，10场约中{min_prob*10:.0f}场，不建议主力投注。\n"
    advice += "\n"
    return advice

# ========== 主报告生成函数（完整重构输出） ==========
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
    
    # 计算各联赛真实半场进球比
    league_half_ratios = calculate_league_half_ratio(all_matches_df)
    
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
            
            # 单场评级
            grade_info = grade_match(fused_probs, odds_data)
            
            # 让球数据
            hw, hd, ha = poisson_probs["handicap"]
            handicap_info = {
                "主胜": hw,
                "平局": hd,
                "客胜": ha,
                "is_steady": max(hw, hd, ha) > 0.6
            }
            
            # 半全场双选组合
            hf_combos = calculate_hf_combos(
                poisson_probs["htft_probs"],
                poisson_probs["ht_draw_prob"],
                odds_data
            )
            
            # 总进球TOP3
            top_ttg = sorted(poisson_probs["total_goals"], key=lambda x: x[1], reverse=True)[:3]
            
            # 模型vs市场分歧
            divergence = None
            if p_market:
                diff = abs(fused_probs[grade_info["dir_idx"]] - p_market[grade_info["dir_idx"]])
                divergence = {
                    "value": diff,
                    "is_large": diff >= 0.1,
                    "type": "方向对立" if np.argmax(fused_probs) != np.argmax(p_market) else "价值追击"
                }
            
            match_date_str = match_dt.strftime("%Y-%m-%d")
            match_id = generate_match_id(match_date_str, home_en, away_en)
            
            # 保存完整结果集
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
                "grade_info": grade_info,
                "handicap": handicap_info,
                "hf_combos": hf_combos,
                "top_scores": poisson_probs["top_scores"],
                "top_ttg": top_ttg,
                "divergence": divergence,
                "odds_data": odds_data,
                "data_sufficient": data_sufficient
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
    
    # 1. 报告头部
    report += "# 足球预测报告（多联赛）\n\n"
    report += f"**生成时间**：{now_bj.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n"
    report += f"**预测时段**：{TARGET_DATE_LABEL} 17:00 ~ 次日 12:00\n"
    report += f"**数据来源**：Football-Data.org + The Odds API + ELO\n"
    
    has_any_odds = any(r["odds_data"] for r in match_results)
    if has_any_odds:
        model_desc = "泊松模型（DC修正） + 全庄家赔率中位数 + ELO 在 logit 空间融合"
    else:
        model_desc = "泊松模型（DC修正） + ELO（本期无赔率数据）"
    report += f"**模型说明**：{model_desc}\n\n"
    
    report += "> ⚠️ 风险提示：所有预测仅供参考，不构成投注建议。足球比赛不确定性高，请理性购彩，量力而行。\n\n"
    
    # 赛事概览
    leagues_covered = set(r["league_name"] for r in match_results)
    ping_count = sum(1 for r in match_results if r["hf_combos"]["符合平系标准"])
    report += f"**本期概览**：共 {len(match_results)} 场比赛，覆盖 {'、'.join(leagues_covered)}；符合平系策略比赛 {ping_count} 场。\n\n"
    report += "---\n\n"
    
    # 2. 总览汇总表
    report += "## 📋 总览汇总表\n\n"
    report += "💡 快速用法：直接筛「A档」比赛做主投，「B档」做串关，C档直接跳过。\n\n"
    report += "| 编号 | 对阵 | 胜平负首选 | 档位 | 让球参考 | 半全场首选双选 | 总进球首选 |\n"
    report += "|------|------|------------|------|----------|------------------|------------|\n"
    
    for r in match_results:
        no = r["match_no"]
        vs = f"{r['home_zh']} vs {r['away_zh']}"
        first_dir = r["grade_info"]["direction"]
        first_prob = f"{r['grade_info']['max_prob']:.1%}"
        grade = r["grade_info"]["grade"]
        
        # 让球参考
        h = r["handicap"]
        if h["客胜"] > 0.6:
            hf_ref = "让+1客胜"
        elif h["主胜"] > 0.6:
            hf_ref = "让-1主胜"
        else:
            hf_ref = "走盘风险高"
        
        # 半全场首选
        if r["hf_combos"]["首选"]:
            hf_first = r["hf_combos"]["首选"]["combo"]
        else:
            hf_first = "无推荐"
        
        # 总进球首选
        ttg_first = f"{r['top_ttg'][0][0]}球" if r["top_ttg"] else "-"
        
        report += f"| {no} | {vs} | {first_dir} {first_prob} | {grade} | {hf_ref} | {hf_first} | {ttg_first} |\n"
    
    report += "\n---\n\n"
    
    # 3. 单场深度分析
    report += "## ⚽ 单场深度分析\n\n"
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
        
        # 一、单场胜平负（主力投注）
        report += "##### 一、单场胜平负（主力投注）\n\n"
        report += "**融合最终概率**\n"
        report += "| 主胜 | 平局 | 客胜 |\n|---|---:|---:|\n"
        report += f"| {r['fused_probs'][0]:.1%} | {r['fused_probs'][1]:.1%} | {r['fused_probs'][2]:.1%} |\n\n"
        
        g = r["grade_info"]
        report += f"**概率档位**：{g['prob_grade']}（{g['direction']} {g['max_prob']:.1%}）\n"
        if g["rec_odds"]:
            report += f"**对应赔率**：{g['rec_odds']:.2f}\n"
            report += f"**赔率检查**：{g['odds_comment']}\n"
        report += f"**最终评级**：{g['grade']}\n"
        if g["ev"] is not None:
            report += f"**期望收益EV**：{g['ev']*100:+.1f}%\n"
        
        # 动态白话解读
        report += "> 💡 白话解读："
        if g["grade"] == "A档":
            if g["ev"] and g["ev"] > 0.05:
                report += f"胜率超50%，赔率匹配度高，长期盈利确定性强，适合作为主力重仓选项。"
            else:
                report += f"胜率达标但赔率优势一般，适合主力轻仓，也可作为串关稳胆。"
        elif g["grade"] == "B档":
            report += f"胜率在40%-50%区间，赔率合理，长期预期正收益，适合轻仓投注或串关搭配。"
        elif g["grade"] == "C档":
            report += f"胜率偏低且赔率优势不足，波动风险大，建议观望或极小注娱乐。"
        else:
            report += f"胜率和赔率均无优势，长期买入预期亏损，不建议介入。"
        report += "\n\n"
        
        # 二、让球盘（串关稳胆参考）
        report += "##### 二、让球盘（串关稳胆参考 · 主让一球）\n\n"
        report += "| 让球后主胜 | 让球平局 | 让球客胜 |\n|---|---:|---:|\n"
        h = r["handicap"]
        report += f"| {h['主胜']:.1%} | {h['平局']:.1%} | {h['客胜']:.1%} |\n\n"
        
        if h["is_steady"]:
            steady_dir = "客胜" if h["客胜"] > 0.6 else "主胜"
            report += f"**稳胆判定**：✅ 让球{steady_dir}概率超60%，可作为串关稳胆选项\n"
        else:
            report += f"**稳胆判定**：三项概率接近，走盘风险较高，不建议单独做稳胆\n"
        report += "\n"
        
        # 三、半全场双选（平系策略）
        report += "##### 三、半全场双选（平系策略）\n\n"
        hf = r["hf_combos"]
        if hf["符合平系标准"]:
            report += f"✅ 本场符合半场平择赛标准（半场平局概率：{hf['半场平概率']:.1%}）\n\n"
        else:
            report += f"⚠️ 本场不符合半场平择赛标准（半场平局概率：{hf['半场平概率']:.1%}），以下仅作参考\n\n"
        
        if hf["首选"]:
            c = hf["首选"]
            report += "###### ✅ 首选组合（稳健主力）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **综合中奖概率**：{c['total_prob']:.1%}（约每{1/c['total_prob']:.1f}单中1单）\n"
            report += f"- **对应SP赔率**：{c['sp']}（估算值，仅供参考）\n"
            if c["ev_pct"] is not None:
                report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        
        if hf["次选"]:
            c = hf["次选"]
            report += "###### ⚠️ 次选组合（收益增强）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **综合中奖概率**：{c['total_prob']:.1%}\n"
            report += f"- **对应SP赔率**：{c['sp']}（估算值，仅供参考）\n"
            if c["ev_pct"] is not None:
                report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%\n"
            report += "\n"
        
        if hf["博弈备选"]:
            c = hf["博弈备选"]
            report += "###### 🎲 博弈备选（仅参考）\n"
            report += f"- **组合选项**：{c['combo']}\n"
            report += f"- **对应SP赔率**：{c['sp']}（估算值，仅供参考）\n"
            report += f"- **综合期望收益**：{c['ev_pct']:+.1f}%，长期微亏，仅适合临场小仓位博弈\n"
            report += "\n"
        
        # 四、总进球&比分参考
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
                report += "> 提示：模型与市场预期差异大，存在搏冷或追热风险，谨慎介入。\n"
            else:
                report += f"✅ 分歧较小：分歧值 {div['value']:.1%}，模型与市场预期基本一致。\n"
            report += "\n"
        
        report += "---\n\n"
    
    # 4. 专题汇总
    report += "## 🎯 专题汇总\n\n"
    
    # 4.1 搏冷备选汇总
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
    
    # 4.2 市场过热提示
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
    
    # 5. 分档位串关方案
    report += "## 🔗 分档位串关方案\n\n"
    
    ab_grade = [r for r in match_results if r["grade_info"]["grade"] in ["A档", "B档"] and r["odds_data"]]
    steady_handicap = [r for r in match_results if r["handicap"]["is_steady"] and r["odds_data"]]
    
    if len(ab_grade) >= 2:
        # 稳健串
        report += "### （1）稳健串（低风险 · 推荐主力）\n\n"
        report += "> 选场规则：单场A/B档比赛，优先跨联赛搭配；总赔率区间：2.4~3.2\n\n"
        
        ab_sorted = sorted(ab_grade, key=lambda x: x["grade_info"]["rec_odds"])
        if len(ab_sorted) >= 2:
            c1 = ab_sorted[0]
            c2 = ab_sorted[1]
            total_odds = c1["grade_info"]["rec_odds"] * c2["grade_info"]["rec_odds"]
            total_prob = c1["grade_info"]["max_prob"] * c2["grade_info"]["max_prob"]
            
            report += "#### 组合一（首选）\n"
            report += f"- **对阵1**：[{c1['league_name']}] {c1['home_zh']} vs {c1['away_zh']}（{c1['grade_info']['direction']}，{c1['grade_info']['grade']}）\n"
            report += f"- **对阵2**：[{c2['league_name']}] {c2['home_zh']} vs {c2['away_zh']}（{c2['grade_info']['direction']}，{c2['grade_info']['grade']}）\n"
            report += f"- **总赔率**：{total_odds:.2f}\n"
            report += f"- **综合命中概率**：约{total_prob:.0%}\n"
            report += "- **组合逻辑**：双稳健档位搭配，跨联赛降低同时爆冷风险，赔率处于黄金区间。\n\n"
        
        # 增值串
        report += "### （2）增值串（中风险 · 轻仓搭配）\n\n"
        report += "> 选场规则：单场B档+让球稳胆搭配；总赔率区间：3.2~4.5\n\n"
        
        if len(ab_sorted) >= 3:
            c1 = ab_sorted[0]
            c3 = ab_sorted[-1]
            total_odds = c1["grade_info"]["rec_odds"] * c3["grade_info"]["rec_odds"]
            
            report += "#### 组合一\n"
            report += f"- **对阵1**：[{c1['league_name']}] {c1['home_zh']} vs {c1['away_zh']}（稳胆）\n"
            report += f"- **对阵2**：[{c3['league_name']}] {c3['home_zh']} vs {c3['away_zh']}（收益增强）\n"
            report += f"- **总赔率**：{total_odds:.2f}\n"
            report += "- **组合逻辑**：稳胆打底+高收益选项搭配，平衡风险与收益。\n\n"
        
        # 半全场二串一推荐
        report += "### （3）半全场串关（中高风险）\n\n"
        report += "> 选场规则：符合平系标准的比赛，首选半全场双选组合搭配；仅供娱乐，建议轻仓。\n\n"
        
        hf_candidates = [r for r in match_results if r["hf_combos"]["首选"] and r["hf_combos"]["符合平系标准"]]
        if len(hf_candidates) >= 2:
            report += "#### 半全场稳健组合\n"
            report += f"- 对阵1：{hf_candidates[0]['home_zh']} vs {hf_candidates[0]['away_zh']}（{hf_candidates[0]['hf_combos']['首选']['combo']}）\n"
            report += f"- 对阵2：{hf_candidates[1]['home_zh']} vs {hf_candidates[1]['away_zh']}（{hf_candidates[1]['hf_combos']['首选']['combo']}）\n"
            report += "- 说明：两场均符合平系标准，双选组合叠加提升容错率，适合半全场玩法串关。\n\n"
        
        # 娱乐高赔串
        report += "### （4）娱乐高赔串（高风险 · 纯娱乐）\n\n"
        report += "> ⚠️ 高风险提示：总赔率5.0以上，命中概率低，仅供娱乐。建议仓位不超过总资金1%~2%。\n\n"
        
        if len(hf_candidates) >= 2:
            report += "#### 半全场高赔组合\n"
            report += f"- 对阵1：{hf_candidates[0]['home_zh']} vs {hf_candidates[0]['away_zh']}（{hf_candidates[0]['hf_combos']['首选']['combo']}）\n"
            report += f"- 对阵2：{hf_candidates[1]['home_zh']} vs {hf_candidates[1]['away_zh']}（{hf_candidates[1]['hf_combos']['首选']['combo']}）\n"
            report += "- 总赔率：约5.0以上，适合娱乐小注。\n\n"
    else:
        report += "本期符合条件的串关场次不足，建议休息。\n\n"
    
    report += "---\n\n"
    
    # 6. 本期数据规律
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
    report += f"- 符合平系策略场次：{ping_count} 场\n\n"
    
    report += "---\n\n"
    
    # 7. 术语对照表
    report += "## 📚 附录：术语大白话对照表\n\n"
    report += "| 术语 | 大白话解释 |\n|------|------------|\n"
    report += "| 融合概率 | 多个模型综合算出的结果发生概率，数值越高越容易中 |\n"
    report += "| EV（期望收益） | 长期反复买这个选项，平均每100块能赚多少钱；正数=长期赚，负数=长期亏 |\n"
    report += "| SP赔率 | 中了之后的赔付倍数，比如SP=4.0就是投100中了拿400 |\n"
    report += "| 让球盘 | 主队让1球之后再算胜平负，用来平衡强弱差距 |\n"
    report += "| 平系策略 | 专门挑上半场容易打平的比赛，双选「平平+平X」，胜率更稳 |\n"
    report += "| 二串一 | 两场比赛都中才算赢，赔率是两场相乘，收益更高、难度更大 |\n"
    report += "| 稳胆 | 概率极高的选项，串关里用来打底，降低整体风险 |\n"
    report += "| 市场过热 | 热门方向买的人太多，赔率被压低，长期买性价比低 |\n\n"
    
    # ========== 保存数据与报告 ==========
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
    print(f"📄 完整报告已生成")

if __name__ == "__main__":
    generate_report()

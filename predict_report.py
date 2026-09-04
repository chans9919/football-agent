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

# ========== 修复问题一：竞彩时间窗口（北京时间 当日17:00 ~ 次日12:00） ==========
def get_time_window():
    now_utc = datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(timezone(timedelta(hours=8)))
    
    # 凌晨 0:00~12:00 仍属于前一天的比赛窗口
    if now_bj.hour < 12:
        base_date = (now_bj - timedelta(days=1)).date()
    else:
        base_date = now_bj.date()
    
    # 组合出完整起止时间（北京时间）
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
    
    # 转回UTC用于匹配API时间（统一为无时区的datetime对象进行比较）
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

# 扩充后的中文名映射
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
    "Luton Town FC": "卢顿",
    "Burnley FC": "伯恩利",
    "Sheffield United FC": "谢菲联",
    "Ipswich Town FC": "伊普斯维奇",
    "Coventry City FC": "考文垂",
    "Sunderland AFC": "桑德兰",
    "Hull City AFC": "赫尔城",

    # 西甲
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
    "1 FC Köln": "科隆",
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

    # 法甲
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
    "Paris FC": "巴黎FC",
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
    
    dc_scores = [(0,0), (1,0), (0,1), (1,1)]
    dc_factor = 1.15
    for i, j in dc_scores:
        if i < matrix.shape[0] and j < matrix.shape[1]:
            matrix[i, j] *= dc_factor
    
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
    draw_prob = 0.28 * (1 - abs(diff) / 600)
    draw_prob = max(0.15, min(0.35, draw_prob))
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


def generate_match_analysis(home_zh, away_zh, lh, la,
                           p_poisson, p_elo, p_market,
                           poisson_probs, home_elo, away_elo,
                           fused_probs, sample_count, data_sufficient):
    analysis = "**详细分析**\n\n"
    
    # 1. 实力定位
    if not data_sufficient:
        analysis += f"- **实力定位**：联赛历史数据严重不足，实力评估基于联赛平均水平，参考价值极低。\n"
    else:
        elo_diff = home_elo - away_elo
        if abs(elo_diff) >= 150:
            level = "实力差距悬殊"
        elif abs(elo_diff) >= 80:
            level = "实力存在明显差距"
        elif abs(elo_diff) >= 30:
            level = "实力有一定差距"
        else:
            level = "实力非常接近"
        
        adv_team = home_zh if elo_diff > 0 else away_zh
        if elo_diff == 0 and home_elo == 1500:
            analysis += f"- **实力定位**：ELO数据暂不充分，参考性有限。"
        else:
            analysis += f"- **实力定位**：两队{level}，ELO综合评分{adv_team}高出{abs(elo_diff):.0f}分。"
        analysis += f"叠加主场优势后，主队理论实力分差为{elo_diff + 65:.0f}分。\n"
    
    # 2. 攻防解读
    avg_goal_ref = 2.6
    total_xg = lh + la
    
    if total_xg > 2.8:
        rhythm = "攻防节奏偏快，大球概率较高"
    elif total_xg < 2.3:
        rhythm = "攻防偏保守，小球概率较高"
    else:
        rhythm = "攻防节奏适中"
    
    analysis += f"- **攻防特点**：双方期望总进球 {total_xg:.2f}，{rhythm}。\n"
    analysis += f"  主队主场期望进球 {lh:.2f}，"
    analysis += f"{'高于' if lh > avg_goal_ref/2 else '低于'}联赛主场平均水平；\n"
    analysis += f"  客队客场期望进球 {la:.2f}，"
    analysis += f"{'高于' if la > avg_goal_ref/2 else '低于'}联赛客场平均水平。\n"
    
    if sample_count < 5:
        analysis += f"  ⚠️ 球队历史样本较少（{sample_count}场），预测波动风险较高。\n"
    
    # 3. 模型一致性
    analysis += "- **模型一致性**："
    directions = []
    
    dir_poisson = np.argmax(p_poisson)
    dir_elo = np.argmax(p_elo)
    directions.append(("泊松", dir_poisson))
    directions.append(("ELO", dir_elo))
    if p_market:
        dir_market = np.argmax(p_market)
        directions.append(("市场", dir_market))
    
    dir_map = {0: "主胜", 1: "平局", 2: "客胜"}
    same_count = sum(1 for _, d in directions if d == directions[0][1])
    
    if same_count == len(directions):
        analysis += f"所有模型方向完全一致，统一看好{dir_map[directions[0][1]]}，分歧度极低。\n"
    else:
        dir_detail = "、".join([f"{name}看{dir_map[d]}" for name, d in directions])
        analysis += f"模型存在一定分歧：{dir_detail}，融合后取折中结果。\n"
    
    # 4. 比分与进球逻辑
    top_score, top_prob = poisson_probs["top_scores"][0]
    peak_goals = max(poisson_probs["total_goals"], key=lambda x: x[1])[0]
    
    analysis += f"- **比分逻辑**：最可能比分 {top_score}（概率{top_prob:.1%}），"
    analysis += f"总进球峰值为 {peak_goals} 球。\n"
    analysis += f"  Top3比分均集中在{'2球及以内' if peak_goals <= 2 else '2球以上'}，"
    analysis += f"属于{'低比分缠斗' if peak_goals <= 2 else '对攻格局'}。\n"
    
    # 5. 半全场走势
    top_htft, top_htft_prob = poisson_probs["top_htft"][0]
    analysis += f"- **半全场走势**：最高概率为 {top_htft}（{top_htft_prob:.1%}），"
    if top_htft.startswith("平"):
        analysis += "上半场平局概率大，比赛慢热、开局谨慎的特征明显。\n"
    else:
        analysis += "上半场大概率分出胜负，开局节奏较快。\n"
    
    # 6. 让球视角
    hw, hd, ha = poisson_probs["handicap"]
    analysis += "- **让球视角（主让一球）**："
    if ha > 0.5:
        analysis += f"让球后客胜概率{ha:.1%}，主队让球偏深，穿盘难度较大。\n"
    elif hw > 0.4:
        analysis += f"让球后主胜概率{hw:.1%}，主队赢球赢盘能力较强。\n"
    else:
        analysis += f"让球后三项概率接近，走盘风险较高。\n"
    
    # 7. 置信度说明
    if not data_sufficient:
        analysis += f"- **置信度说明**：联赛基础数据不足，预测参考价值极低，仅供参考。\n"
    else:
        max_prob = max(fused_probs[0], fused_probs[1], fused_probs[2])
        if max_prob >= 0.5:
            confidence = "较高"
        elif max_prob >= 0.4:
            confidence = "中等"
        else:
            confidence = "一般"
        
        analysis += f"- **置信度说明**：融合模型最大概率{max_prob:.1%}，信心{confidence}。"
        if not p_market:
            analysis += "（缺少市场赔率校准，参考性略有下降）"
        analysis += "\n"
    
    return analysis


def generate_report():
    if not os.path.exists("data/matches.csv"):
        print("❌ 没有找到历史数据文件 data/matches.csv")
        return

    # 加载赔率
    odds_df = pd.DataFrame()
    if os.path.exists("data/odds.csv"):
        odds_df = pd.read_csv("data/odds.csv")
        print(f"📊 加载赔率数据：{len(odds_df)} 条")
    else:
        print("⚠️ 未找到赔率文件")

    # 拉取未来赛事
    all_upcoming = []
    for league in LEAGUES:
        matches = get_upcoming_matches(league, days_ahead=2)
        all_upcoming.extend(matches)

    # ========== 修复问题三：显式UTC解析再去时区，安全比较 ==========
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

    # 预加载全部历史数据
    all_matches_df = pd.read_csv("data/matches.csv")
    print(f"📚 历史数据总场数：{len(all_matches_df)}")
    if "league" in all_matches_df.columns:
        print("🏷️ 历史数据包含league字段，各联赛数量：")
        print(all_matches_df["league"].value_counts().to_string())
    else:
        print("⚠️ 历史数据没有league字段，默认全部按英超处理")

    # 按联赛缓存ELO和数据
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

    # ========== 修复问题四：删除冗余变量，直接判断是否有赔率 ==========
    has_any_odds = False
    for match in all_upcoming:
        odds_data = find_odds(odds_df, match["home_team"], match["away_team"])
        if odds_data:
            has_any_odds = True
            break
    
    if has_any_odds:
        model_desc = "泊松模型（DC修正） + 全庄家赔率中位数 + ELO 在 logit 空间融合"
    else:
        model_desc = "泊松模型（DC修正） + ELO（本期无赔率数据）"

    # ========== 修复问题二：生成时间使用真实北京时间 ==========
    now_bj = datetime.utcnow() + timedelta(hours=8)
    report = "# 足球预测报告（多联赛）\n\n"
    report += f"生成时间：{now_bj.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n\n"
    report += f"预测时段：{TARGET_DATE_LABEL} 17:00 ~ 次日 12:00\n\n"
    report += f"数据来源：Football-Data.org + The Odds API + ELO\n\n"
    report += f"模型说明：{model_desc}\n\n"
    report += f"共 {len(all_upcoming)} 场比赛\n\n---\n\n"

    # 按开赛时间排序
    all_upcoming_sorted = sorted(all_upcoming, key=lambda x: x["date"])
    current_league = ""

    for match in all_upcoming_sorted:
        league = match["league"]
        league_df = league_df_cache.get(league, pd.DataFrame())
        elo_dict = elo_cache.get(league, {})
        data_sufficient = data_sufficient_cache.get(league, False)
        league_name = LEAGUE_NAMES.get(league, league)

        # 按联赛加分隔标题
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

            # 时区转换
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

            # 融合权重
            if p_market:
                weights = [0.4, 0.4, 0.2]
                probs_list = [p_poisson, p_market, p_elo]
            else:
                weights = [0.6, 0.2]
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

            # 估算样本量
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

            # 比赛标题
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

                        rec_zh = {"home": "主胜", "draw": "平局", "away": "客胜"}[pred_direction]
            report += f"**推荐方向**：{rec_zh}（置信度 {conf:.1%}）\n\n"
            if odds_data:
                ev_home = fused_home * odds_data[0] - 1
                ev_draw = fused_draw * odds_data[1] - 1
                ev_away = fused_away * odds_data[2] - 1
                evs = {"主胜": ev_home, "平局": ev_draw, "客胜": ev_away}
                best_bet = max(evs, key=evs.get)
                best_ev = evs[best_bet]
                if best_ev > 0.15:
                    grade = "A 档（强推）"
                elif best_ev > 0.05:
                    grade = "B 档（可买）"
                elif best_ev > 0:
                    grade = "C 档（观望）"
                else:
                    grade = "不建议下注"
                report += f"**期望价值（EV）**\n"
                report += f"| 选项 | 模型概率 | 赔率 | EV |\n|---|---:|---:|---:|\n"
                report += f"| 主胜 | {fused_home:.1%} | {odds_data[0]:.2f} | {ev_home:+.1%} |\n"
                report += f"| 平局 | {fused_draw:.1%} | {odds_data[1]:.2f} | {ev_draw:+.1%} |\n"
                report += f"| 客胜 | {fused_away:.1%} | {odds_data[2]:.2f} | {ev_away:+.1%} |\n\n"
                report += f"**投注建议**：{best_bet} {grade}（EV {best_ev:+.1%}）\n\n"
            else:
                report += f"**投注建议**：无赔率数据，无法计算期望价值\n\n"

            report += f"**比分 Top3**\n"
            for score, prob in poisson_probs['top_scores']:
                report += f"- {score}（{prob:.1%}）\n"
            report += "\n"
            report += f"**半全场 Top3**\n"
            for htft, prob in poisson_probs['top_htft']:
                report += f"- {htft}（{prob:.1%}）\n"
            report += "\n"
            report += f"**让球（主让一球）**\n"
            hw, hd, ha = poisson_probs['handicap']
            report += f"| 让球后主胜 | 让球平局 | 让球客胜 |\n|---|---:|---:|\n"
            report += f"| {hw:.1%} | {hd:.1%} | {ha:.1%} |\n\n"
            report += f"**总进球分布**\n"
            for total, prob in poisson_probs['total_goals'][:5]:
                report += f"- {total}球：{prob:.1%}  "
            report += "\n\n"

            # 详细分析
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

    print(f"\n✅ 成功生成 {match_counter-1} 场比赛预测")

    # 保存预测记录
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

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from scipy.stats import poisson
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

# ================== 球队中文名映射 ==================
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
    "Deportivo Alavés": "阿拉维斯",
    "UD Las Palmas": "拉斯帕尔马斯",
    "CD Leganés": "莱加内斯",
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
    "Holstein Kiel": "基尔",
    "FC St. Pauli": "圣保利",
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
    "Parma Calcio 1913": "帕尔马",
    "Como 1907": "科莫",
    "Venezia FC": "威尼斯",
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
    "AS Saint-Étienne": "圣埃蒂安",
    "Stade Malherbe Caen": "卡昂",
    "FC Girondins de Bordeaux": "波尔多",
    "ESTAC Troyes": "特鲁瓦",
    "Dijon FCO": "第戎",
}

def get_team_name_zh(team_en):
    return TEAM_NAMES_ZH.get(team_en, team_en)

# ================== 工具函数 ==================
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

def poisson_prob_matrix(lambda_home, lambda_away, max_goals=8):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    matrix /= matrix.sum()
    return matrix

def match_probabilities(lambda_home, lambda_away):
    """返回胜平负、比分Top3、半全场Top3、让球概率、总进球分布"""
    matrix = poisson_prob_matrix(lambda_home, lambda_away)
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    score_probs = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            score_probs[f"{i}-{j}"] = matrix[i,j]
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    # 半全场（简化独立）
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
    # 让球（主让一球）
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
    # 总进球分布
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

def calculate_elo(df, initial=1500):
    """根据历史比赛数据计算每支球队的当前 ELO 评分。K 因子自适应。"""
    elo = {}
    games_played = {}
    df = df.sort_values("date")
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_goals"]
        ag = row["away_goals"]
        if home not in elo:
            elo[home] = initial
            games_played[home] = 0
        if away not in elo:
            elo[away] = initial
            games_played[away] = 0
        k_home = 40 if games_played[home] < 10 else 25
        k_away = 40 if games_played[away] < 10 else 25
        rating_home = elo[home]
        rating_away = elo[away]
        exp_home = 1 / (1 + 10 ** ((rating_away - rating_home) / 400))
        exp_away = 1 - exp_home
        if hg > ag:
            actual_home = 1
            actual_away = 0
        elif hg == ag:
            actual_home = 0.5
            actual_away = 0.5
        else:
            actual_home = 0
            actual_away = 1
        elo[home] = rating_home + k_home * (actual_home - exp_home)
        elo[away] = rating_away + k_away * (actual_away - exp_away)
        games_played[home] += 1
        games_played[away] += 1
    return elo

def elo_probabilities(home_elo, away_elo, home_adv=65):
    """根据 ELO 差计算主胜/平/客胜概率。平局概率动态化。"""
    diff = home_elo + home_adv - away_elo
    draw_prob = 0.28 * (1 - abs(diff) / 500)
    draw_prob = max(0.15, min(0.35, draw_prob))
    expected_home_win = 1 / (1 + 10 ** (-diff / 400))
    remaining = 1 - draw_prob
    home_win = expected_home_win * remaining
    away_win = (1 - expected_home_win) * remaining
    total = home_win + draw_prob + away_win
    return home_win/total, draw_prob/total, away_win/total

def find_odds(odds_df, home_en, away_en):
    """在赔率数据中查找匹配的比赛，返回 (odds_home, odds_draw, odds_away) 或 None"""
    if odds_df.empty:
        return None
    match = odds_df[(odds_df["home_team"] == home_en) & (odds_df["away_team"] == away_en)]
    if not match.empty:
        row = match.iloc[0]
        return row["odds_home"], row["odds_draw"], row["odds_away"]
    for _, row in odds_df.iterrows():
        if (home_en.lower() in row["home_team"].lower() or row["home_team"].lower() in home_en.lower()) and \
           (away_en.lower() in row["away_team"].lower() or row["away_team"].lower() in away_en.lower()):
            return row["odds_home"], row["odds_draw"], row["odds_away"]
    return None

def generate_match_id(date_str, home_team, away_team):
    """生成唯一 match_id：日期 + 标准化主队名 + 标准化客队名"""
    def normalize(name):
        return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
    return f"{date_str}_{normalize(home_team)}_{normalize(away_team)}"

# ================== 主报告生成函数 ==================
def generate_report():
    if not os.path.exists("data/matches.csv"):
        return "没有历史数据文件 data/matches.csv"
    df = pd.read_csv("data/matches.csv")
    odds_df = pd.DataFrame()
    if os.path.exists("data/odds.csv"):
        odds_df = pd.read_csv("data/odds.csv")
    
    elo_dict = calculate_elo(df)
    
    all_upcoming = []
    for league in LEAGUES:
        matches = get_upcoming_matches(league, days_ahead=3)
        all_upcoming.extend(matches)
    
    if not all_upcoming:
        return "未来 3 天内没有找到任何联赛的未开始比赛"
    
    report = "# 足球预测报告（多联赛）\n\n"
    report += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n\n"
    report += f"数据来源：Football-Data.org + The Odds API + ELO\n\n"
    report += f"模型说明：泊松模型（权重0.4）+ 市场赔率（权重0.4）+ ELO（权重0.2）融合；ELO平局概率动态化\n\n"
    report += f"共 {len(all_upcoming)} 场比赛\n\n---\n\n"
    
    match_counter = 1
    predictions = []  # 用于存储结构化预测结果
    
    for league in LEAGUES:
        league_matches = [m for m in all_upcoming if m["league"] == league]
        if not league_matches:
            continue
        league_name = LEAGUE_NAMES.get(league, league)
        report += f"## {league_name}\n\n"
        for match in league_matches:
            home_en = match["home_team"]
            away_en = match["away_team"]
            home_zh = get_team_name_zh(home_en)
            away_zh = get_team_name_zh(away_en)
            
            try:
                match_time = pd.to_datetime(match["date"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
            except:
                match_time = match["date"]
            
            # 从历史数据中筛选该联赛
            league_df = df[df["league"] == league]
            if league_df.empty:
                league_df = df
            try:
                lh, la = train_poisson(league_df, home_en, away_en)
            except Exception as e:
                print(f"计算 {home_en} vs {away_en} 出错：{e}")
                lh, la = 1.5, 1.2
            
            # 泊松概率
            poisson_probs = match_probabilities(lh, la)
            p_poisson = [poisson_probs["home_win"], poisson_probs["draw"], poisson_probs["away_win"]]
            
            # 市场概率（已除水）
            odds_data = find_odds(odds_df, home_en, away_en)
            if odds_data:
                odds_home, odds_draw, odds_away = odds_data
                raw_probs = [1/odds_home, 1/odds_draw, 1/odds_away]
                total_raw = sum(raw_probs)
                p_market = [p / total_raw for p in raw_probs]
            else:
                p_market = None
            
            # ELO 概率
            home_elo = elo_dict.get(home_en, 1500)
            away_elo = elo_dict.get(away_en, 1500)
            p_elo = elo_probabilities(home_elo, away_elo)
            
            # 融合
            if p_market:
                weights = [0.4, 0.4, 0.2]
                final_probs = (
                    weights[0] * np.array(p_poisson) +
                    weights[1] * np.array(p_market) +
                    weights[2] * np.array(p_elo)
                )
            else:
                weights_no_market = [0.6, 0.2]
                final_probs = (
                    weights_no_market[0] * np.array(p_poisson) +
                    weights_no_market[1] * np.array(p_elo)
                )
                final_probs /= weights_no_market[0] + weights_no_market[1]
            final_probs /= final_probs.sum()
            
            # 确定推荐方向
            if final_probs[0] >= final_probs[1] and final_probs[0] >= final_probs[2]:
                pred_direction = "home"
                conf = final_probs[0]
            elif final_probs[1] >= final_probs[0] and final_probs[1] >= final_probs[2]:
                pred_direction = "draw"
                conf = final_probs[1]
            else:
                pred_direction = "away"
                conf = final_probs[2]
            
            # 生成 match_id
            match_date_str = pd.to_datetime(match["date"]).strftime("%Y-%m-%d")
            match_id = generate_match_id(match_date_str, home_en, away_en)
            
            # 构建预测记录
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
                "fused_home": final_probs[0],
                "fused_draw": final_probs[1],
                "fused_away": final_probs[2],
                "pred_direction": pred_direction,
                "status": "pending"
            }
            predictions.append(record)
            
            # ========== 报告写入 ==========
            match_no = f"{match_counter:03d}"
            match_counter += 1
            
            report += f"### {match_no} {home_zh} vs {away_zh}\n\n"
            report += f"- 联赛：{league_name}\n"
            report += f"- 时间：{match_time}\n"
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
            report += f"| {final_probs[0]:.1%} | {final_probs[1]:.1%} | {final_probs[2]:.1%} |\n\n"
            
            if pred_direction == "home":
                rec_zh = "主胜"
            elif pred_direction == "draw":
                rec_zh = "平局"
            else:
                rec_zh = "客胜"
            report += f"**推荐方向**：{rec_zh}（置信度 {conf:.1%}）\n\n"
            
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
            report += f"| 让球后主胜 | 让球后平局 | 让球后客胜 |\n|---|---:|---:|\n"
            report += f"| {hw:.1%} | {hd:.1%} | {ha:.1%} |\n\n"
            
            report += f"**总进球分布**\n"
            for total, prob in poisson_probs['total_goals'][:5]:
                report += f"- {total}球：{prob:.1%}  "
            report += "\n\n---\n\n"
    
    # ========== 保存预测结果 ==========
    if predictions:
        new_pred_df = pd.DataFrame(predictions)
        os.makedirs("data", exist_ok=True)
        # 合并旧的历史记录（status = finished）
        if os.path.exists("data/predictions.csv"):
            old_preds = pd.read_csv("data/predictions.csv")
            old_finished = old_preds[old_preds["status"] == "finished"]
            combined = pd.concat([old_finished, new_pred_df], ignore_index=True)
            combined.to_csv("data/predictions.csv", index=False)
            print(f"已保存 {len(combined)} 条预测记录（含历史已结算 {len(old_finished)} 条）")
        else:
            new_pred_df.to_csv("data/predictions.csv", index=False)
            print(f"已保存 {len(new_pred_df)} 条预测记录")
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已生成，共 {len(all_upcoming)} 场比赛")

if __name__ == "__main__":
    generate_report()

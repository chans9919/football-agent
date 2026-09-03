import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime, timedelta
from scipy.stats import poisson
from train_model import train_poisson, predict_match_prob
from team_config import normalize_team_name

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

# 中文名映射
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
    return TEAM_NAMES_ZH.get(team_en, team_en)


def get_upcoming_matches(league_code, days_ahead=3):
    if not API_KEY:
        return []
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
        return upcoming
    except Exception as e:
        print(f"⚠️ 联赛 {league_code} 未来赛事请求失败: {e}")
        return []


def poisson_prob_matrix(lambda_home, lambda_away, max_goals=8):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    # DC 简化修正
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

    # 半全场
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

    # 让球
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
        row = match.iloc[0]
        return row["odds_home"], row["odds_draw"], row["odds_away"]
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


def load_elo(league):
    path = f"model/elo_{league}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_report():
    if not os.path.exists("data/matches.csv"):
        print("没有历史数据文件")
        return

    # 加载赔率
    odds_df = pd.DataFrame()
    if os.path.exists("data/odds.csv"):
        odds_df = pd.read_csv("data/odds.csv")

    # 加载所有未来比赛
    all_upcoming = []
    for league in LEAGUES:
        matches = get_upcoming_matches(league, days_ahead=3)
        all_upcoming.extend(matches)

    if not all_upcoming:
        print("未来3天无比赛")
        return

    report = "# 足球预测报告（多联赛）\n\n"
    report += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n\n"
    report += f"数据来源：Football-Data.org + The Odds API + ELO\n\n"
    report += f"模型说明：泊松模型 + 市场赔率 + ELO 在 logit 空间融合\n\n"
    report += f"共 {len(all_upcoming)} 场比赛\n\n---\n\n"

    match_counter = 1
    predictions = []

    for league in LEAGUES:
        league_matches = [m for m in all_upcoming if m["league"] == league]
        if not league_matches:
            continue

        # 加载该联赛的模型数据
        league_df = pd.read_csv("data/matches.csv")
        league_df = league_df[league_df["league"] == league].copy() if "league" in league_df.columns else pd.DataFrame()
        if len(league_df) < 50:
            print(f"⏭️ {league} 历史数据不足，跳过预测")
            continue

        elo_dict = load_elo(league)
        league_name = LEAGUE_NAMES.get(league, league)
        report += f"## {league_name}\n\n"

        for match in league_matches:
            try:
                home_en = match["home_team"]
                away_en = match["away_team"]
                home_zh = get_team_name_zh(home_en)
                away_zh = get_team_name_zh(away_en)

                # 修复时区转换
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
                    "status": "pending"
                }
                predictions.append(record)

                # 报告输出
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
                report += f"| {fused_home:.1%} | {fused_draw:.1%} | {fused_away:.1%} |\n\n"

                rec_zh = {"home": "主胜", "draw": "平局", "away": "客胜"}[pred_direction]
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

            except Exception as e:
                print(f"跳过比赛 {match.get('home_team', '?')} vs {match.get('away_team', '?')}: {e}")
                continue

    # 【修复】保存预测记录：保留所有历史记录，按match_id去重
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
        print(f"✅ 预测记录已保存，共 {len(combined)} 条")

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 报告已生成，共 {len(all_upcoming)} 场比赛")


if __name__ == "__main__":
    generate_report()

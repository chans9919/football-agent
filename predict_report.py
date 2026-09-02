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

# ================== 球队中文名映射（大幅扩充） ==================
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
    "FC Nantes": "南特",
    "AS Saint-Étienne": "圣埃蒂安",
    "Stade Malherbe Caen": "卡昂",
    "FC Girondins de Bordeaux": "波尔多",
    "FC Lorient": "洛里昂",
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
    """计算泊松比分矩阵"""
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    matrix /= matrix.sum()  # 归一化
    return matrix

def match_probabilities(lambda_home, lambda_away):
    """返回胜平负概率、比分Top3、半全场Top3、让球概率、总进球分布"""
    matrix = poisson_prob_matrix(lambda_home, lambda_away)
    # 胜平负
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    # 比分Top3
    score_probs = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            score_probs[f"{i}-{j}"] = matrix[i,j]
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    # 半全场：上半场（前45分钟）可近似为全场的一半时间，lambda 减半
    lambda_half_home = lambda_home * 0.45  # 45/90
    lambda_half_away = lambda_away * 0.45
    half_matrix = poisson_prob_matrix(lambda_half_home, lambda_half_away, max_goals=4)
    ht_home = np.sum(np.tril(half_matrix, -1))
    ht_draw = np.sum(np.diag(half_matrix))
    ht_away = np.sum(np.triu(half_matrix, 1))
    # 半全场组合概率（简化：假设半场和全场独立，实际可更精确，但暂用独立近似）
    ft_home = home_win
    ft_draw = draw
    ft_away = away_win
    htft_probs = {
        "胜胜": ht_home * ft_home,
        "胜平": ht_home * ft_draw,
        "胜负": ht_home * ft_away,
        "平胜": ht_draw * ft_home,
        "平平": ht_draw * ft_draw,
        "平负": ht_draw * ft_away,
        "负胜": ht_away * ft_home,
        "负平": ht_away * ft_draw,
        "负负": ht_away * ft_away,
    }
    top_htft = sorted(htft_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    # 让球（主让一球 -1）：即主队进球-1后与客队比较
    # 重新计算让球后的胜平负
    matrix_handicap = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # 主队让1球：主队实际进球i，视为i-1
            new_i = i - 1
            if new_i < 0:
                new_i = -1  # 表示主队落后1球以上
            # 简单处理：用原始矩阵，判断新比分
            if new_i == -1:
                # 主队相当于-1，必定客胜
                if j >= 0:
                    matrix_handicap[i,j] = matrix[i,j]  # 客胜
            else:
                if new_i > j:
                    # 主胜（让球后）
                    matrix_handicap[i,j] = matrix[i,j]
                elif new_i == j:
                    # 平局
                    matrix_handicap[i,j] = matrix[i,j]
                else:
                    # 客胜
                    matrix_handicap[i,j] = matrix[i,j]
    # 重新归类
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

def generate_report():
    if not os.path.exists("data/matches.csv"):
        return "没有历史数据文件 data/matches.csv"
    df = pd.read_csv("data/matches.csv")
    
    all_upcoming = []
    for league in LEAGUES:
        matches = get_upcoming_matches(league, days_ahead=3)
        all_upcoming.extend(matches)
    
    if not all_upcoming:
        return "未来 3 天内没有找到任何联赛的未开始比赛"
    
    report = "# 足球预测报告（多联赛）\n\n"
    report += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（北京时间）\n\n"
    report += f"数据来源：Football-Data.org 免费 API\n\n"
    report += f"模型说明：泊松分布模型，基于历史 100 天数据训练，平滑因子 10\n\n"
    report += f"共 {len(all_upcoming)} 场比赛\n\n---\n\n"
    
    # 按联赛分组输出，并给每场比赛编号
    match_counter = 1
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
            
            # 比赛时间
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
            
            # 计算所有概率
            probs = match_probabilities(lh, la)
            
            # 编号格式：三位数
            match_no = f"{match_counter:03d}"
            match_counter += 1
            
            report += f"### {match_no} {home_zh} vs {away_zh}\n\n"
            report += f"- 联赛：{league_name}\n"
            report += f"- 时间：{match_time}\n"
            report += f"- 期望进球：主 {lh:.2f}，客 {la:.2f}\n\n"
            report += f"**胜平负概率**\n"
            report += f"| 主胜 | 平局 | 客胜 |\n|---|---|---|\n"
            report += f"| {probs['home_win']:.1%} | {probs['draw']:.1%} | {probs['away_win']:.1%} |\n\n"
            
            report += f"**比分 Top3**\n"
            for score, prob in probs['top_scores']:
                report += f"- {score}（{prob:.1%}）\n"
            report += "\n"
            
            report += f"**半全场 Top3**\n"
            for htft, prob in probs['top_htft']:
                report += f"- {htft}（{prob:.1%}）\n"
            report += "\n"
            
            report += f"**让球（主让一球）**\n"
            hw, hd, ha = probs['handicap']
            report += f"| 让球后主胜 | 让球后平局 | 让球后客胜 |\n|---|---|---|\n"
            report += f"| {hw:.1%} | {hd:.1%} | {ha:.1%} |\n\n"
            
            report += f"**总进球分布**\n"
            for total, prob in probs['total_goals'][:5]:
                report += f"- {total}球：{prob:.1%}  "
            report += "\n\n---\n\n"
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已生成，共 {len(all_upcoming)} 场比赛")

if __name__ == "__main__":
    generate_report()

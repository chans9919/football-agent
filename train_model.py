import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import log_loss
import json
import os


def load_data(league=None):
    path = "data/matches.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    if league and "league" in df.columns:
        df = df[df["league"] == league].copy()
    
    # 统一清洗
    df = df.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df[(df["home_goals"] >= 0) & (df["away_goals"] >= 0)]
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def get_initial_elo(team_name):
    top_teams = {
        "Manchester City FC": 1800,
        "Arsenal FC": 1750,
        "Liverpool FC": 1780,
        "Chelsea FC": 1700,
        "Manchester United FC": 1720,
        "Real Madrid CF": 1850,
        "FC Barcelona": 1830,
        "FC Bayern München": 1820,
        "Juventus FC": 1750,
        "Paris Saint-Germain FC": 1780,
    }
    weak_teams = {
        "Luton Town FC": 1400,
        "Burnley FC": 1420,
        "Sheffield United FC": 1380,
    }
    if team_name in top_teams:
        return top_teams[team_name]
    elif team_name in weak_teams:
        return weak_teams[team_name]
    else:
        return 1500


def calculate_elo(df):
    elo = {}
    games_played = {}
    df = df.sort_values("date")

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_goals"]
        ag = row["away_goals"]

        if home not in elo:
            elo[home] = get_initial_elo(home)
            games_played[home] = 0
        if away not in elo:
            elo[away] = get_initial_elo(away)
            games_played[away] = 0

        k_home = 40 if games_played[home] < 10 else 25
        k_away = 40 if games_played[away] < 10 else 25

        rating_home = elo[home]
        rating_away = elo[away]
        exp_home = 1 / (1 + 10 ** ((rating_away - rating_home) / 400))

        if hg > ag:
            actual_home = 1
        elif hg == ag:
            actual_home = 0.5
        else:
            actual_home = 0

        # ========== 修复问题一：客队使用自身k_away ==========
        elo[home] = rating_home + k_home * (actual_home - exp_home)
        elo[away] = rating_away - k_away * (actual_home - exp_home)
        
        games_played[home] += 1
        games_played[away] += 1

    return elo


def train_poisson(df, home_team, away_team):
    if df.empty:
        return 1.5, 1.2

    avg_goals = (df["home_goals"].mean() + df["away_goals"].mean()) / 2
    if avg_goals == 0 or pd.isna(avg_goals):
        avg_goals = 1.5

    home_matches = df[df["home_team"] == home_team]
    away_matches = df[df["away_team"] == away_team]

    if len(home_matches) > 0:
        home_attack_raw = home_matches["home_goals"].mean()
        home_def_raw = home_matches["away_goals"].mean()
    else:
        home_attack_raw = avg_goals
        home_def_raw = avg_goals

    if len(away_matches) > 0:
        away_attack_raw = away_matches["away_goals"].mean()
        away_def_raw = away_matches["home_goals"].mean()
    else:
        away_attack_raw = avg_goals
        away_def_raw = avg_goals

    def dynamic_smoothing(games_count):
        if games_count <= 5:
            return 20.0
        elif games_count <= 10:
            return 15.0
        else:
            return 10.0

    home_games_count = len(home_matches)
    away_games_count = len(away_matches)
    smoothing_home = dynamic_smoothing(home_games_count)
    smoothing_away = dynamic_smoothing(away_games_count)

    weight_home = home_games_count / (home_games_count + smoothing_home)
    weight_away = away_games_count / (away_games_count + smoothing_away)

    home_attack = (home_attack_raw / avg_goals) * weight_home + 1.0 * (1 - weight_home)
    home_def    = (home_def_raw / avg_goals) * weight_home + 1.0 * (1 - weight_home)
    away_attack = (away_attack_raw / avg_goals) * weight_away + 1.0 * (1 - weight_away)
    away_def    = (away_def_raw / avg_goals) * weight_away + 1.0 * (1 - weight_away)

    home_attack = np.clip(home_attack, 0.3, 3.0)
    home_def    = np.clip(home_def, 0.3, 3.0)
    away_attack = np.clip(away_attack, 0.3, 3.0)
    away_def    = np.clip(away_def, 0.3, 3.0)

    lambda_home = avg_goals * home_attack * away_def
    lambda_away = avg_goals * away_attack * home_def

    lambda_home = np.clip(lambda_home, 0.2, 6.0)
    lambda_away = np.clip(lambda_away, 0.2, 6.0)

    return lambda_home, lambda_away


# ========== 修复问题三：统一使用带DC修正的概率计算 ==========
def poisson_prob_matrix(lambda_home, lambda_away, max_goals=10):
    """和predict_report共用同一套DC修正逻辑，保证训练评估一致"""
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    
    # Dixon-Coles 简化修正
    dc_scores = [(0,0), (1,0), (0,1), (1,1)]
    dc_factor = 1.15
    for i, j in dc_scores:
        if i < matrix.shape[0] and j < matrix.shape[1]:
            matrix[i, j] *= dc_factor
    
    matrix /= matrix.sum()
    return matrix


def predict_match_prob(lambda_home, lambda_away):
    matrix = poisson_prob_matrix(lambda_home, lambda_away)
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    return home_win, draw, away_win


def evaluate_model(df, test_ratio=0.2):
    if len(df) < 20:
        return None

    df = df.sort_values("date")
    split = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        lh, la = train_poisson(train_df, row["home_team"], row["away_team"])
        ph, pd_, pa = predict_match_prob(lh, la)
        y_pred.append([ph, pd_, pa])
        if row["home_goals"] > row["away_goals"]:
            y_true.append(0)
        elif row["home_goals"] == row["away_goals"]:
            y_true.append(1)
        else:
            y_true.append(2)
    return log_loss(y_true, y_pred)


def save_model(elo_dict, metrics, league="PL"):
    os.makedirs("model", exist_ok=True)
    
    # 保存ELO评分
    with open(f"model/elo_{league}.json", "w", encoding="utf-8") as f:
        json.dump(elo_dict, f, ensure_ascii=False, indent=2)
    
    # 保存评估指标
    with open(f"model/metrics_{league}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    leagues = ["PL", "PD", "BL1", "SA", "FL1"]
    
    for league in leagues:
        df = load_data(league=league)
        if len(df) < 50:
            print(f"⏭️ {league} 数据不足，跳过训练")
            continue
        
        print(f"\n===== 训练联赛: {league} =====")
        print(f"比赛场数: {len(df)}")
        
        # 计算ELO
        elo_dict = calculate_elo(df)
        print(f"ELO计算完成，共 {len(elo_dict)} 支球队")
        
        # 评估模型（带DC修正，和预测端一致）
        current_loss = evaluate_model(df)
        print(f"模型log_loss: {current_loss:.4f}")
        
        metrics = {
            "matches": len(df),
            "log_loss": current_loss,
            "updated": pd.Timestamp.now().isoformat()
        }
        
        save_model(elo_dict, metrics, league=league)
        print(f"✅ {league} 模型参数已保存")

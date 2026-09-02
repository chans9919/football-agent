import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import log_loss
import json
import os

def load_data():
    if os.path.exists("data/matches.csv"):
        df = pd.read_csv("data/matches.csv")
        return df
    else:
        return pd.DataFrame()

def train_poisson(df, home_team, away_team):
    """
    训练泊松模型，返回 lambda_home 和 lambda_away。
    使用动态贝叶斯平滑，避免小样本极端值。
    """
    if df.empty:
        return 1.5, 1.2

    avg_goals = (df["home_goals"].mean() + df["away_goals"].mean()) / 2
    if avg_goals == 0 or pd.isna(avg_goals):
        avg_goals = 1.5

    home_matches = df[df["home_team"] == home_team]
    away_matches = df[df["away_team"] == away_team]

    # 原始攻防因子（正确版本）
    if len(home_matches) > 0:
        home_attack_raw = home_matches["home_goals"].mean()  # 主队主场进球
        home_def_raw = home_matches["away_goals"].mean()     # 主队主场失球（客队进球）
    else:
        home_attack_raw = avg_goals
        home_def_raw = avg_goals

    if len(away_matches) > 0:
        away_attack_raw = away_matches["away_goals"].mean()  # 客队客场进球
        away_def_raw = away_matches["home_goals"].mean()     # 客队客场失球（主队进球）
    else:
        away_attack_raw = avg_goals
        away_def_raw = avg_goals

    # 动态平滑系数：比赛场次越少，收缩越强
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

    # 贝叶斯平滑：向联赛均值（1.0）收缩
    home_attack = (home_attack_raw / avg_goals) * weight_home + 1.0 * (1 - weight_home)
    home_def    = (home_def_raw / avg_goals) * weight_home + 1.0 * (1 - weight_home)
    away_attack = (away_attack_raw / avg_goals) * weight_away + 1.0 * (1 - weight_away)
    away_def    = (away_def_raw / avg_goals) * weight_away + 1.0 * (1 - weight_away)

    # 限制因子范围，防止极端值
    home_attack = np.clip(home_attack, 0.3, 3.0)
    home_def    = np.clip(home_def, 0.3, 3.0)
    away_attack = np.clip(away_attack, 0.3, 3.0)
    away_def    = np.clip(away_def, 0.3, 3.0)

    lambda_home = avg_goals * home_attack * away_def
    lambda_away = avg_goals * away_attack * home_def

    lambda_home = np.clip(lambda_home, 0.2, 6.0)
    lambda_away = np.clip(lambda_away, 0.2, 6.0)

    return lambda_home, lambda_away

def predict_match_prob(lambda_home, lambda_away):
    max_goals = 10
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    matrix_sum = matrix.sum()
    if matrix_sum > 0:
        matrix /= matrix_sum
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

def save_model_metrics(metrics, path="model/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    df = load_data()
    if df.empty:
        print("No data")
        exit()
    
    current_loss = evaluate_model(df)
    print(f"Current model log loss: {current_loss}")
    
    old_loss = None
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f:
            old_metrics = json.load(f)
            old_loss = old_metrics.get("log_loss")
    
    if old_loss is None or (current_loss is not None and current_loss < old_loss):
        metrics = {"log_loss": current_loss, "updated": pd.Timestamp.now().isoformat()}
        save_model_metrics(metrics)
        print("Model updated")
    else:
        print("Model not updated, keeping previous")

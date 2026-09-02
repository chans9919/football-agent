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

def train_poisson(df, home_team, away_team, smoothing=10.0):
    """
    训练泊松模型，返回 lambda_home 和 lambda_away。
    使用平滑处理，避免因子为 0 导致的极端概率。
    """
    if df.empty:
        return 1.5, 1.2  # 默认值

    avg_goals = (df["home_goals"].mean() + df["away_goals"].mean()) / 2
    if avg_goals == 0 or pd.isna(avg_goals):
        avg_goals = 1.5  # 兜底

    # 球队作为主队的比赛数
    home_matches = df[df["home_team"] == home_team]
    away_matches = df[df["away_team"] == away_team]
    
    # 主队进攻因子（作为主队时的场均进球）
    if len(home_matches) > 0:
        home_attack_raw = home_matches["home_goals"].mean()
    else:
        home_attack_raw = avg_goals  # 无数据时用联赛平均

    # 主队防守因子（作为主队时的场均失球）
    if len(home_matches) > 0:
        home_def_raw = home_matches["away_goals"].mean()
    else:
        home_def_raw = avg_goals

    # 客队进攻因子（作为客队时的场均进球）
    if len(away_matches) > 0:
        away_attack_raw = away_matches["away_goals"].mean()
    else:
        away_attack_raw = avg_goals

    # 客队防守因子（作为客队时的场均失球）
    if len(away_matches) > 0:
        away_def_raw = away_matches["home_goals"].mean()
    else:
        away_def_raw = avg_goals

    # 平滑：加权平均（比赛场次越多，越信任实际值）
    home_games_count = len(home_matches)
    away_games_count = len(away_matches)
    weight_home = home_games_count / (home_games_count + smoothing)
    weight_away = away_games_count / (away_games_count + smoothing)

    home_attack = (home_attack_raw / avg_goals) * weight_home + 1.0 * (1 - weight_home)
    home_def    = (home_def_raw / avg_goals) * weight_home + 1.0 * (1 - weight_home)
    away_attack = (away_attack_raw / avg_goals) * weight_away + 1.0 * (1 - weight_away)
    away_def    = (away_def_raw / avg_goals) * weight_away + 1.0 * (1 - weight_away)

    # 确保因子非负且合理（限制在 0.3 ~ 3.0 之间）
    home_attack = np.clip(home_attack, 0.3, 3.0)
    home_def    = np.clip(home_def, 0.3, 3.0)
    away_attack = np.clip(away_attack, 0.3, 3.0)
    away_def    = np.clip(away_def, 0.3, 3.0)

    lambda_home = avg_goals * home_attack * away_def
    lambda_away = avg_goals * away_attack * home_def

    # 限制 lambda 在合理范围
    lambda_home = np.clip(lambda_home, 0.2, 6.0)
    lambda_away = np.clip(lambda_away, 0.2, 6.0)

    return lambda_home, lambda_away

def predict_match_prob(lambda_home, lambda_away):
    max_goals = 10
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    # 归一化（避免截断导致的概率和不足 1）
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
    
    y_true = []   # 存储类别索引：0=主胜，1=平局，2=客胜
    y_pred = []   # 存储概率向量 [p_home, p_draw, p_away]
    
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
    
    # 计算 log loss，不再传入 labels 参数
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

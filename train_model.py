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
    """训练泊松模型，返回 lambda_home 和 lambda_away"""
    # 简化：使用所有比赛的平均进球作为基准，并计算主客队进攻/防守因子
    avg_goals = (df["home_goals"].mean() + df["away_goals"].mean()) / 2
    # 进攻因子 = 球队场均进球 / 联赛平均
    # 防守因子 = 球队场均失球 / 联赛平均
    # 为避免数据稀疏，使用简单平均
    home_attack = df[df["home_team"] == home_team]["home_goals"].mean() / avg_goals if len(df[df["home_team"] == home_team]) > 0 else 1.0
    away_attack = df[df["away_team"] == away_team]["away_goals"].mean() / avg_goals if len(df[df["away_team"] == away_team]) > 0 else 1.0
    home_def = df[df["home_team"] == home_team]["away_goals"].mean() / avg_goals if len(df[df["home_team"] == home_team]) > 0 else 1.0
    away_def = df[df["away_team"] == away_team]["home_goals"].mean() / avg_goals if len(df[df["away_team"] == away_team]) > 0 else 1.0
    
    lambda_home = avg_goals * home_attack * away_def
    lambda_away = avg_goals * away_attack * home_def
    return lambda_home, lambda_away

def predict_match_prob(lambda_home, lambda_away):
    max_goals = 10
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i,j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    return home_win, draw, away_win

def evaluate_model(df, test_ratio=0.2):
    if len(df) < 20:
        return None
    # 简单按时间划分，最后20%作为测试
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
            y_true.append([1,0,0])
        elif row["home_goals"] == row["away_goals"]:
            y_true.append([0,1,0])
        else:
            y_true.append([0,0,1])
    return log_loss(y_true, y_pred, labels=[0,1,2])

def save_model_metrics(metrics, path="model/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    df = load_data()
    if df.empty:
        print("No data")
        exit()
    
    # 计算当前模型在最近数据上的 log loss
    current_loss = evaluate_model(df)
    print(f"Current model log loss: {current_loss}")
    
    # 检查是否有旧模型指标
    old_loss = None
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f:
            old_metrics = json.load(f)
            old_loss = old_metrics.get("log_loss")
    
    # 如果当前模型更好（log loss 更低），或者没有旧模型，则保存
    if old_loss is None or (current_loss is not None and current_loss < old_loss):
        metrics = {"log_loss": current_loss, "updated": pd.Timestamp.now().isoformat()}
        save_model_metrics(metrics)
        print("Model updated")
    else:
        print("Model not updated, keeping previous")

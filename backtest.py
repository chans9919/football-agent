import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_predictions():
    if os.path.exists("data/predictions.csv"):
        return pd.read_csv("data/predictions.csv")
    return pd.DataFrame()

def load_matches():
    if os.path.exists("data/matches.csv"):
        return pd.read_csv("data/matches.csv")
    return pd.DataFrame()

def generate_match_id(date_str, home_team, away_team):
    def normalize(name):
        return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
    return f"{date_str}_{normalize(home_team)}_{normalize(away_team)}"

def backtest():
    preds = load_predictions()
    matches = load_matches()
    if preds.empty or matches.empty:
        print("缺少预测或比赛数据，无法回测")
        return
    
    # 只处理状态为 pending 的预测
    pending = preds[preds["status"] == "pending"].copy()
    if pending.empty:
        print("没有待结算的预测")
        return
    
    history = []
    if os.path.exists("data/history.csv"):
        history_df = pd.read_csv("data/history.csv")
        history = history_df.to_dict("records")
    
    for _, pred in pending.iterrows():
        match_date = pred["date"]
        home = pred["home_team"]
        away = pred["away_team"]
        # 在历史比赛数据中查找对应比赛
        result = matches[(matches["date"] == match_date) & 
                         (matches["home_team"] == home) & 
                         (matches["away_team"] == away)]
        if result.empty:
            continue  # 比赛还没结束或没匹配上
        
        actual = result.iloc[0]
        home_goals = actual["home_goals"]
        away_goals = actual["away_goals"]
        if home_goals > away_goals:
            actual_direction = "home"
        elif home_goals == away_goals:
            actual_direction = "draw"
        else:
            actual_direction = "away"
        
        # 计算对数损失（简化，防止 log(0) 用极小值）
        def log_loss_single(actual_dir, p_home, p_draw, p_away):
            if actual_dir == "home":
                return -np.log(max(p_home, 1e-10))
            elif actual_dir == "draw":
                return -np.log(max(p_draw, 1e-10))
            else:
                return -np.log(max(p_away, 1e-10))
        
        poisson_loss = log_loss_single(actual_direction, pred["poisson_home"], pred["poisson_draw"], pred["poisson_away"])
        if not np.isnan(pred["market_home"]):
            market_loss = log_loss_single(actual_direction, pred["market_home"], pred["market_draw"], pred["market_away"])
        else:
            market_loss = np.nan
        elo_loss = log_loss_single(actual_direction, pred["elo_home"], pred["elo_draw"], pred["elo_away"])
        fused_loss = log_loss_single(actual_direction, pred["fused_home"], pred["fused_draw"], pred["fused_away"])
        
        # 构建历史记录
        record = {
            "match_id": pred["match_id"],
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "actual_direction": actual_direction,
            "pred_direction": pred["pred_direction"],
            "correct": 1 if actual_direction == pred["pred_direction"] else 0,
            "poisson_loss": poisson_loss,
            "market_loss": market_loss,
            "elo_loss": elo_loss,
            "fused_loss": fused_loss,
        }
        history.append(record)
        # 更新预测状态为 finished
        preds.loc[preds["match_id"] == pred["match_id"], "status"] = "finished"
    
    # 保存更新后的 predictions.csv
    preds.to_csv("data/predictions.csv", index=False)
    
    # 保存回测历史记录
    if history:
        hist_df = pd.DataFrame(history)
        os.makedirs("data", exist_ok=True)
        hist_df.to_csv("data/history.csv", index=False)
        print(f"回测完成，共结算 {len(hist_df)} 场比赛")
    else:
        print("没有新的结算")

if __name__ == "__main__":
    backtest()

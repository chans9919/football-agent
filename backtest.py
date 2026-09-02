import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import log_loss, brier_score_loss

def main():
    pred_path = "data/predictions.csv"
    match_path = "data/matches.csv"
    if not os.path.exists(pred_path) or not os.path.exists(match_path):
        print("数据不足，跳过回测")
        return

    preds = pd.read_csv(pred_path)
    matches = pd.read_csv(match_path)

    pending = preds[preds["status"] == "pending"].copy()
    if len(pending) == 0:
        print("没有待结算的预测")
        return

    correct = 0
    y_true_list = []
    y_pred_list = []
    settled_ids = []

    for _, row in pending.iterrows():
        actual = matches[(matches["date"] == row["date"]) & 
                         (matches["home_team"] == row["home_team"]) & 
                         (matches["away_team"] == row["away_team"])]
        if len(actual) == 0:
            continue
        actual_row = actual.iloc[0]
        hg = actual_row["home_goals"]
        ag = actual_row["away_goals"]
        if hg > ag:
            true_label = 0
        elif hg == ag:
            true_label = 1
        else:
            true_label = 2

        pred_probs = [row["fused_home"], row["fused_draw"], row["fused_away"]]
        pred_label = np.argmax(pred_probs)
        if pred_label == true_label:
            correct += 1

        y_true_list.append(true_label)
        y_pred_list.append(pred_probs)
        settled_ids.append(row["match_id"])

    if len(y_true_list) == 0:
        print("没有可结算的比赛")
        return

    # 更新状态
    for mid in settled_ids:
        preds.loc[preds["match_id"] == mid, "status"] = "finished"
    preds.to_csv(pred_path, index=False)

    accuracy = correct / len(y_true_list)
    logloss = log_loss(y_true_list, y_pred_list, labels=[0,1,2])
    # Brier 分数计算（三分类，需要 one-hot 编码）
    y_true_onehot = np.zeros((len(y_true_list), 3))
    for i, label in enumerate(y_true_list):
        y_true_onehot[i, label] = 1
    brier = np.mean(np.sum((np.array(y_pred_list) - y_true_onehot) ** 2, axis=1))

    history = pd.DataFrame([{
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "matches": len(y_true_list),
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier_score": brier
    }])
    hist_path = "data/history.csv"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(hist_path):
        history.to_csv(hist_path, mode="a", header=False, index=False)
    else:
        history.to_csv(hist_path, index=False)

    print(f"回测完成：{len(y_true_list)}场，命中率{accuracy:.1%}，log_loss {logloss:.4f}，Brier {brier:.4f}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.metrics import log_loss


def calculate_brier(y_true, y_pred):
    y_true_onehot = np.zeros((len(y_true), 3))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    return np.mean(np.sum((np.array(y_pred) - y_true_onehot) ** 2, axis=1))


def main():
    pred_path = "data/predictions.csv"
    match_path = "data/matches.csv"

    if not os.path.exists(pred_path) or not os.path.exists(match_path):
        print("数据不足，跳过回测")
        return

    preds = pd.read_csv(pred_path)
    matches = pd.read_csv(match_path)

    # 只处理待结算的预测
    pending = preds[preds["status"] == "pending"].copy()
    if len(pending) == 0:
        print("没有待结算的预测")
        return

    # 过期清理：超过7天还没匹配的标记为expired
    cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    expired_mask = pending["date"] < cutoff_date
    if expired_mask.any():
        preds.loc[preds["match_id"].isin(pending[expired_mask]["match_id"]), "status"] = "expired"
        pending = pending[~expired_mask]
        print(f"🗑️ 清理过期预测 {expired_mask.sum()} 条")

    if len(pending) == 0:
        preds.to_csv(pred_path, index=False)
        return

    settled_ids = []
    results = []

    for _, row in pending.iterrows():
        # 匹配实际结果
        actual = matches[
            (matches["date"] == row["date"]) & 
            (matches["league"] == row["league"]) &
            (matches["home_team"] == row["home_team"]) & 
            (matches["away_team"] == row["away_team"])
        ]
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

        # 收集各模型概率
        models = {
            "poisson": [row["poisson_home"], row["poisson_draw"], row["poisson_away"]],
            "elo": [row["elo_home"], row["elo_draw"], row["elo_away"]],
            "fused": [row["fused_home"], row["fused_draw"], row["fused_away"]],
        }
        if not pd.isna(row.get("market_home", np.nan)):
            models["market"] = [row["market_home"], row["market_draw"], row["market_away"]]

        results.append({
            "league": row["league"],
            "true_label": true_label,
            **models
        })
        settled_ids.append(row["match_id"])

    if len(results) == 0:
        print("没有可结算的比赛")
        preds.to_csv(pred_path, index=False)
        return

    # 更新状态
    preds.loc[preds["match_id"].isin(settled_ids), "status"] = "finished"
    preds.to_csv(pred_path, index=False)

    # ========== 分模型统计 ==========
    print("\n" + "="*50)
    print(f"📊 回测完成：共结算 {len(results)} 场比赛")
    print("="*50)

    result_df = pd.DataFrame(results)
    model_names = [c for c in result_df.columns if c not in ["league", "true_label"]]

    for model in model_names:
        y_true = result_df["true_label"].tolist()
        y_pred = result_df[model].tolist()
        
        correct = sum(1 for t, p in zip(y_true, y_pred) if np.argmax(p) == t)
        acc = correct / len(y_true)
        logloss = log_loss(y_true, y_pred, labels=[0,1,2])
        brier = calculate_brier(y_true, y_pred)
        
        print(f"\n【{model.upper()} 模型】")
        print(f"  命中率: {acc:.1%}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")

    # ========== 分联赛统计（融合模型） ==========
    print("\n" + "-"*50)
    print("🏆 分联赛表现（融合模型）")
    for league in result_df["league"].unique():
        league_df = result_df[result_df["league"] == league]
        y_true = league_df["true_label"].tolist()
        y_pred = league_df["fused"].tolist()
        correct = sum(1 for t, p in zip(y_true, y_pred) if np.argmax(p) == t)
        acc = correct / len(y_true)
        print(f"  {league}: {len(league_df)}场，命中率{acc:.1%}")

    # 保存历史记录
    history = pd.DataFrame([{
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "matches": len(results),
        "accuracy": np.mean([np.argmax(p) == t for t, p in zip(result_df["true_label"], result_df["fused"])]),
        "log_loss": log_loss(result_df["true_label"], result_df["fused"].tolist(), labels=[0,1,2]),
        "brier_score": calculate_brier(result_df["true_label"], result_df["fused"].tolist())
    }])

    hist_path = "data/history.csv"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(hist_path):
        history.to_csv(hist_path, mode="a", header=False, index=False)
    else:
        history.to_csv(hist_path, index=False)

    print("\n" + "="*50)


if __name__ == "__main__":
    main()

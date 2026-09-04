# backtest_historical.py
# 用历史数据做回测，验证泊松+ELO模型的预测能力
# 用法：python backtest_historical.py

import pandas as pd
import numpy as np
from train_model import train_poisson, calculate_elo
from predict_report import poisson_prob_matrix, elo_probabilities
from sklearn.metrics import log_loss

def main():
    df = pd.read_csv("data/matches.csv")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"总数据量: {len(df)} 场\n")

    for league in ["PL", "PD", "BL1", "SA", "FL1"]:
        league_df = df[df["league"] == league].copy().reset_index(drop=True)
        if len(league_df) < 100:
            print(f"{league}: 数据不足 ({len(league_df)}场)，跳过")
            continue

        # 前80%训练，后20%测试
        split = int(len(league_df) * 0.8)
        test_df = league_df.iloc[split:]
        print(f"{'='*50}")
        print(f"{league}: 总{len(league_df)}场，训练{split}场，测试{len(test_df)}场")

        # 计算ELO（用全部历史数据算到训练结束时的状态）
        train_df = league_df.iloc[:split]
        elo_dict = calculate_elo(train_df)

        # 泊松回测
        y_true = []
        y_pred_poisson = []
        y_pred_elo = []
        y_pred_fused = []
        correct_poisson = 0
        correct_elo = 0
        correct_fused = 0

        for _, row in test_df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hg = row["home_goals"]
            ag = row["away_goals"]

            # 真实结果
            if hg > ag:
                true_label = 0
            elif hg == ag:
                true_label = 1
            else:
                true_label = 2

            # 泊松预测
            lh, la = train_poisson(train_df, home, away)
            matrix = poisson_prob_matrix(lh, la)
            p_poisson = [
                np.sum(np.tril(matrix, -1)),
                np.sum(np.diag(matrix)),
                np.sum(np.triu(matrix, 1))
            ]

            # ELO预测
            home_elo = elo_dict.get(home, 1500)
            away_elo = elo_dict.get(away, 1500)
            p_elo = list(elo_probabilities(home_elo, away_elo))

            # 融合（泊松0.6 + ELO0.2，归一化后 0.75:0.25）
            from predict_report import logit, inv_logit
            weights = [0.75, 0.25]
            fused = []
            for i in range(3):
                logits = [logit(p_poisson[i]), logit(p_elo[i])]
                fused_val = sum(w * l for w, l in zip(weights, logits))
                fused.append(inv_logit(fused_val))
            total = sum(fused)
            fused = [f / total for f in fused]

            y_true.append(true_label)
            y_pred_poisson.append(p_poisson)
            y_pred_elo.append(p_elo)
            y_pred_fused.append(fused)

            if np.argmax(p_poisson) == true_label:
                correct_poisson += 1
            if np.argmax(p_elo) == true_label:
                correct_elo += 1
            if np.argmax(fused) == true_label:
                correct_fused += 1

            # 更新ELO
            k = 30
            exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            if hg > ag:
                actual = 1
            elif hg == ag:
                actual = 0.5
            else:
                actual = 0
            elo_dict[home] = home_elo + k * (actual - exp_home)
            elo_dict[away] = away_elo - k * (actual - exp_home)

            # 更新训练集（滑动窗口）
            train_df = pd.concat([train_df, row.to_frame().T], ignore_index=True)

        n = len(test_df)
        print(f"\n--- {league} 回测结果 ({n}场) ---")
        print(f"泊松模型:  命中 {correct_poisson}/{n} = {correct_poisson/n:.1%}")
        print(f"ELO模型:   命中 {correct_elo}/{n} = {correct_elo/n:.1%}")
        print(f"融合模型:  命中 {correct_fused}/{n} = {correct_fused/n:.1%}")
        print(f"泊松 log_loss: {log_loss(y_true, y_pred_poisson, labels=[0,1,2]):.4f}")
        print(f"ELO  log_loss: {log_loss(y_true, y_pred_elo, labels=[0,1,2]):.4f}")
        print(f"融合 log_loss: {log_loss(y_true, y_pred_fused, labels=[0,1,2]):.4f}")

        # 分结果统计
        for label, name in [(0, "主胜"), (1, "平局"), (2, "客胜")]:
            total_count = sum(1 for t in y_true if t == label)
            correct_count = sum(1 for t, p in zip(y_true, y_pred_fused) if t == label and np.argmax(p) == label)
            if total_count > 0:
                print(f"  {name}: {correct_count}/{total_count} = {correct_count/total_count:.1%}")

    print(f"\n{'='*50}")
    print("回测完成")


if __name__ == "__main__":
    main()

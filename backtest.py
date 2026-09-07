import os
from datetime import datetime, date
from collections import defaultdict
from data_fetcher import load_matches_csv, load_odds_csv
from predict_core import TeamRating, match_probability_matrix, extract_win_draw_loss, calculate_half_full_probs, get_pingxi_double_selection, calibrate_pingxi, select_pingxi_matches

def run_backtest(start_date: date = None, end_date: date = None):
    all_matches = load_matches_csv()
    
    if start_date:
        all_matches = [m for m in all_matches if m["date"] >= start_date]
    if end_date:
        all_matches = [m for m in all_matches if m["date"] <= end_date]
    
    if len(all_matches) < 10:
        print("数据量不足，无法回测")
        return
    
    rating = TeamRating()
    results = []
    
    # 按天滑动回测
    dates = sorted(set(m["date"] for m in all_matches))
    
    for d in dates:
        day_matches = [m for m in all_matches if m["date"] == d]
        
        # 用当天之前的数据预测
        for m in day_matches:
            home_xg, away_xg = rating.get_expected_goals(m["home_team"], m["away_team"])
            matrix = match_probability_matrix(home_xg, away_xg)
            wdl = extract_win_draw_loss(matrix)
            hf = calculate_half_full_probs(home_xg, away_xg, m["league"])
            pingxi = get_pingxi_double_selection(hf["half_draw"], wdl["home_win"], wdl["away_win"], wdl["draw"])
            
            results.append({
                **m,
                "pred_home": wdl["home_win"],
                "pred_draw": wdl["draw"],
                "pred_away": wdl["away_win"],
                "pred_half_draw": hf["half_draw"],
                "calibrated_half_draw": pingxi["calibrated_half_draw"],
                "pingxi_primary": pingxi["primary"],
                "pingxi_secondary": pingxi["secondary"],
                "pingxi_combined": pingxi["combined_prob"]
            })
        
        # 当天比赛结束后更新评分
        for m in day_matches:
            rating.add_match(m["home_team"], m["away_team"], m["home_goals"], m["away_goals"], m["date"])
    
    # ========== 统计输出 ==========
    total = len(results)
    half_valid = [r for r in results if r["has_half_time"]]
    
    # 胜平负命中率
    wdl_correct = 0
    for r in results:
        pred = "home" if r["pred_home"] > r["pred_away"] else "away"
        actual = "home" if r["home_goals"] > r["away_goals"] else ("away" if r["home_goals"] < r["away_goals"] else "draw")
        if pred == actual:
            wdl_correct += 1
    
    # 平系统计（仅有效半场）
    pingxi_all = [r for r in half_valid if r["calibrated_half_draw"] >= 0.38]
    pingxi_hits = 0
    for r in pingxi_all:
        half_draw_actual = (r["half_home_goals"] == r["half_away_goals"])
        full_result = "home" if r["home_goals"] > r["away_goals"] else ("draw" if r["home_goals"] == r["away_goals"] else "away")
        
        if r["pred_home"] >= r["pred_away"]:
            hit = half_draw_actual and full_result in ["home", "draw"]
        else:
            hit = half_draw_actual and full_result in ["away", "draw"]
        
        if hit:
            pingxi_hits += 1
    
    # 输出报告
    lines = []
    lines.append("=" * 50)
    lines.append("回测报告")
    lines.append(f"统计区间：{results[0]['date']} ~ {results[-1]['date']}")
    lines.append(f"总场次：{total}")
    lines.append(f"半场有效场次：{len(half_valid)}")
    lines.append("=" * 50)
    lines.append("")
    lines.append("一、胜平负总体命中率")
    lines.append(f"  {wdl_correct}/{total} = {wdl_correct/total:.1%}")
    lines.append("")
    lines.append("二、平系双选统计（仅有效半场样本）")
    if pingxi_all:
        lines.append(f"  入选场次：{len(pingxi_all)}")
        lines.append(f"  命中场次：{pingxi_hits}")
        lines.append(f"  命中率：{pingxi_hits/len(pingxi_all):.1%}")
    else:
        lines.append("  无符合条件的平系场次")
    lines.append("")
    lines.append("三、半场平局校准统计")
    if half_valid:
        actual_half_draw = sum(1 for r in half_valid if r["half_home_goals"] == r["half_away_goals"])
        lines.append(f"  实际半场平局率：{actual_half_draw/len(half_valid):.1%}")
        avg_pred = sum(r["pred_half_draw"] for r in half_valid) / len(half_valid)
        lines.append(f"  平均预测半场平：{avg_pred:.1%}")
        avg_cal = sum(r["calibrated_half_draw"] for r in half_valid) / len(half_valid)
        lines.append(f"  平均校准后半平：{avg_cal:.1%}")
    lines.append("")
    lines.append("=" * 50)
    
    report = "\n".join(lines)
    print(report)
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/backtest_result.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    
    run_backtest(start, end)

import os
import json
from datetime import datetime, date
from data_fetcher import load_matches_csv, load_odds_csv
from predict_core import TeamRating, match_probability_matrix, extract_win_draw_loss, calculate_half_full_probs, get_pingxi_double_selection, calibrate_pingxi

def settle_date(target_date: date, save: bool = True):
    all_matches = load_matches_csv()
    odds_map = load_odds_csv()
    
    # 训练评分
    rating = TeamRating()
    historical = [m for m in all_matches if m["date"] < target_date]
    for m in historical:
        rating.add_match(m["home_team"], m["away_team"], m["home_goals"], m["away_goals"], m["date"])
    
    today_matches = [m for m in all_matches if m["date"] == target_date]
    if not today_matches:
        print(f"{target_date} 无比赛数据")
        return
    
    total = len(today_matches)
    wdl_correct = 0
    half_valid = [m for m in today_matches if m["has_half_time"]]
    pingxi_hits = 0
    pingxi_count = 0
    
    lines = []
    lines.append(f"每日结算报告 {target_date}")
    lines.append("=" * 40)
    lines.append(f"总场次：{total}")
    lines.append(f"半场有效场次：{len(half_valid)}")
    lines.append("")
    
    for m in today_matches:
        home_xg, away_xg = rating.get_expected_goals(m["home_team"], m["away_team"])
        matrix = match_probability_matrix(home_xg, away_xg)
        wdl = extract_win_draw_loss(matrix)
        hf = calculate_half_full_probs(home_xg, away_xg, m["league"])
        
        # 胜平负命中判断
        pred = "home" if wdl["home_win"] > wdl["away_win"] else "away"
        actual = "home" if m["home_goals"] > m["away_goals"] else ("away" if m["home_goals"] < m["away_goals"] else "draw")
        wdl_hit = (pred == actual)
        if wdl_hit:
            wdl_correct += 1
        
        # 平系命中判断（仅有效半场）
        pingxi_hit = False
        if m["has_half_time"]:
            cal = calibrate_pingxi(hf["half_draw"])
            if cal >= 0.38:
                pingxi_count += 1
                half_draw_actual = (m["half_home_goals"] == m["half_away_goals"])
                full_result = "home" if m["home_goals"] > m["away_goals"] else ("draw" if m["home_goals"] == m["away_goals"] else "away")
                
                if wdl["home_win"] >= wdl["away_win"]:
                    pingxi_hit = half_draw_actual and full_result in ["home", "draw"]
                else:
                    pingxi_hit = half_draw_actual and full_result in ["away", "draw"]
                
                if pingxi_hit:
                    pingxi_hits += 1
        
        lines.append(f"{m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}")
        lines.append(f"  预测主胜{wdl['home_win']:.1%} 客胜{wdl['away_win']:.1%} | 命中：{'✓' if wdl_hit else '✗'}")
        if m["has_half_time"]:
            lines.append(f"  半场平校准后{calibrate_pingxi(hf['half_draw']):.1%} | 平系命中：{'✓' if pingxi_hit else '✗'}")
        lines.append("")
    
    lines.append("-" * 40)
    lines.append(f"胜平负命中率：{wdl_correct}/{total} = {wdl_correct/total:.1%}")
    if pingxi_count > 0:
        lines.append(f"平系命中率：{pingxi_hits}/{pingxi_count} = {pingxi_hits/pingxi_count:.1%}")
    else:
        lines.append("平系场次：0")
    
    report = "\n".join(lines)
    print(report)
    
    if save:
        os.makedirs("settle", exist_ok=True)
        date_str = target_date.strftime("%Y-%m-%d")
        with open(f"settle/settle_{date_str}.txt", "w", encoding="utf-8") as f:
            f.write(report)
    
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="结算日期 YYYY-MM-DD")
    parser.add_argument("--no-save", action="store_true", help="不保存文件")
    args = parser.parse_args()
    
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    settle_date(target_date, save=not args.no_save)

import os
from datetime import datetime, date
from data_fetcher import load_matches_csv
from predict_core import TeamRating, match_probability_matrix, extract_win_draw_loss, calculate_half_full_probs, get_pingxi_double_selection, select_pingxi_matches
from motivation import calculate_motivation
from report_markdown import generate_markdown_report
from report_html import generate_html_report

def generate_report(target_date: date = None, format_type: str = "html"):
    if target_date is None:
        target_date = date.today()
    
    # 加载历史数据
    all_matches = load_matches_csv()
    
    # 初始化评分系统，用目标日期之前的数据训练
    rating = TeamRating()
    historical = [m for m in all_matches if m["date"] < target_date]
    for m in historical:
        rating.add_match(m["home_team"], m["away_team"], m["home_goals"], m["away_goals"], m["date"])
    
    # 取目标日期的比赛
    today_matches = [m for m in all_matches if m["date"] == target_date]
    
    results = []
    for m in today_matches:
        home_xg, away_xg = rating.get_expected_goals(m["home_team"], m["away_team"])
        
        matrix = match_probability_matrix(home_xg, away_xg)
        wdl = extract_win_draw_loss(matrix)
        hf = calculate_half_full_probs(home_xg, away_xg, m["league"])
        
        pingxi = get_pingxi_double_selection(
            hf["half_draw"], wdl["home_win"], wdl["away_win"], wdl["draw"]
        )
        
        results.append({
            **m,
            "home_xg": round(home_xg, 3),
            "away_xg": round(away_xg, 3),
            "home_win": wdl["home_win"],
            "draw": wdl["draw"],
            "away_win": wdl["away_win"],
            "half_draw": hf["half_draw"],
            "calibrated_half_draw": pingxi["calibrated_half_draw"],
            "pingxi_primary": pingxi["primary"],
            "pingxi_secondary": pingxi["secondary"],
            "combined_prob": pingxi["combined_prob"]
        })
    
    pingxi_selected = select_pingxi_matches(results)
    
    # 生成报告
    os.makedirs("reports", exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")
    
    if format_type in ["md", "markdown", "all"]:
        md_content = generate_markdown_report(results, pingxi_selected, target_date)
        with open(f"reports/report_{date_str}.md", "w", encoding="utf-8") as f:
            f.write(md_content)
    
    if format_type in ["html", "all"]:
        html_content = generate_html_report(results, pingxi_selected, target_date)
        with open(f"reports/report_{date_str}.html", "w", encoding="utf-8") as f:
            f.write(html_content)
    
    print(f"报告生成完成：reports/report_{date_str}.{format_type}")
    return results, pingxi_selected

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--format", default="html", help="报告格式: html/md/all")
    args = parser.parse_args()
    
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    generate_report(target_date, args.format)

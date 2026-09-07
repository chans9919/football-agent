import pandas as pd
import os
from datetime import datetime
from team_name_normalizer import normalize_team_name

def fetch_odds_data(matches: list, output_csv: str = "data/odds.csv"):
    """抓取赔率数据，按比赛去重，保留最新记录"""
    # 模拟/示例逻辑：实际使用时替换为真实API
    new_rows = []
    for m in matches:
        new_rows.append({
            "date": m["date"].strftime("%Y-%m-%d"),
            "home_team": normalize_team_name(m["home_team"]),
            "away_team": normalize_team_name(m["away_team"]),
            "home_odds": 1.85,
            "draw_odds": 3.40,
            "away_odds": 4.20,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    new_df = pd.DataFrame(new_rows)
    
    # 读取旧数据并去重
    if os.path.exists(output_csv):
        old_df = pd.read_csv(output_csv, encoding="utf-8-sig")
        combined = pd.concat([new_df, old_df], ignore_index=True)
        # 按key去重，保留最新
        combined["key"] = combined["date"] + "|" + combined["home_team"] + "|" + combined["away_team"]
        combined = combined.sort_values("fetched_at", ascending=False).drop_duplicates("key", keep="first")
        combined = combined.drop("key", axis=1)
    else:
        combined = new_df
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"赔率数据更新完成，共 {len(combined)} 条记录")

if __name__ == "__main__":
    # 示例调用
    from data_fetcher import load_matches_csv
    matches = load_matches_csv()[-10:]
    fetch_odds_data(matches)

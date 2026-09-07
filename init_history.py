import pandas as pd
from team_name_normalizer import normalize_team_name

def init_history_data(source_csv: str, output_csv: str = "data/matches.csv"):
    """初始化历史数据，清洗队名，标记半场有效性"""
    df = pd.read_csv(source_csv)
    
    # 标准化队名
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team_name)
    
    # 半场数据有效性标记
    df["has_half_time"] = df["HTHG"].notna() & df["HTAG"].notna()
    
    # 统一日期格式
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"历史数据初始化完成，共 {len(df)} 场比赛")

if __name__ == "__main__":
    init_history_data("data/football-data-co-uk.csv")

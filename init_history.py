import pandas as pd
import requests
import os
from io import StringIO

# ========== 可配置项 ==========
# 需要加载的历史赛季，格式：两位年份拼接，可自行增减赛季
SEASONS = ["2223", "2324", "2425", "2526"]
# 数据源地址（GitHub Actions服务器可直接访问）
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# 队名统一映射：数据源队名 → 你的标准队名（和实时抓取保持一致）
TEAM_NAME_MAP = {
    "Liverpool": "Liverpool FC",
    "Ipswich": "Ipswich Town FC",
    "Man City": "Manchester City FC",
    "Man United": "Manchester United FC",
    "Arsenal": "Arsenal FC",
    "Chelsea": "Chelsea FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Aston Villa": "Aston Villa FC",
    "Newcastle": "Newcastle United FC",
    "Brighton": "Brighton & Hove Albion FC",
    "West Ham": "West Ham United FC",
    "Brentford": "Brentford FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Fulham": "Fulham FC",
    "Bournemouth": "AFC Bournemouth",
    "Wolves": "Wolverhampton Wanderers FC",
    "Sheffield Utd": "Sheffield United FC",
    "Burnley": "Burnley FC",
    "Luton": "Luton Town FC",
    "Leicester": "Leicester City FC",
    "Southampton": "Southampton FC",
    "Leeds": "Leeds United FC",
}

# ========== 内部处理函数 ==========
def download_season(season):
    """下载单个赛季的原始数据"""
    url = BASE_URL.format(season=season)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # 只读取需要的5个核心字段
        df = pd.read_csv(StringIO(resp.text), usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        return df
    except Exception as e:
        print(f"⚠️ 赛季 {season} 下载失败: {str(e)}")
        return pd.DataFrame()

def process_data(df):
    """数据清洗、格式转换、队名统一"""
    # 重命名为标准字段
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals"
    })
    
    # 日期格式转换：dd/mm/yyyy → yyyy-mm-dd
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    
    # 统一队名
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
    
    # 数据类型校验与清洗
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["home_goals", "away_goals"])
    
    return df

# ========== 主执行流程 ==========
def main():
    all_seasons_data = []
    
    for season in SEASONS:
        print(f"正在处理赛季: {season}")
        raw_df = download_season(season)
        if raw_df.empty:
            continue
        processed_df = process_data(raw_df)
        all_seasons_data.append(processed_df)
        print(f"  ✅ 本赛季加载 {len(processed_df)} 场比赛")
    
    if not all_seasons_data:
        print("❌ 未获取到任何历史数据，流程终止")
        return

    # 合并所有赛季数据
    history_df = pd.concat(all_seasons_data, ignore_index=True)
    
    # 和现有数据合并、按比赛唯一标识去重
    data_path = "data/matches.csv"
    if os.path.exists(data_path):
        existing_df = pd.read_csv(data_path)
        history_df = pd.concat([existing_df, history_df], ignore_index=True)
        # 按「日期+主队+客队」去重，保留最新版本
        history_df = history_df.drop_duplicates(
            subset=["date", "home_team", "away_team"], 
            keep="last"
        )
    
    # 按日期升序排序（ELO模型依赖时间顺序）
    history_df = history_df.sort_values("date").reset_index(drop=True)
    
    # 确保目录存在并保存
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    history_df.to_csv(data_path, index=False, encoding="utf-8")
    
    print("\n" + "="*40)
    print(f"✅ 历史数据初始化完成")
    print(f"总计比赛场数: {len(history_df)}")
    print(f"数据已保存至: {data_path}")
    print("="*40)

if __name__ == "__main__":
    main()

import pandas as pd
import requests
import os
from io import StringIO
from team_config import normalize_team_name

# ========== 可配置项 ==========
# 需要加载的历史赛季，格式：两位年份拼接，可自行增减赛季
SEASONS = ["2223", "2324", "2425", "2526"]

# 联赛配置：数据源编码 → 系统标准编码（和predict_report保持一致）
# 数据源编码参考：E0=英超, SP1=西甲, D1=德甲, I1=意甲, F1=法甲
LEAGUE_MAP = {
    "E0": "PL",   # 英超
    "SP1": "PD",  # 西甲
    "D1": "BL1",   # 德甲
    "I1": "SA",    # 意甲
    "F1": "FL1"    # 法甲
}

# 数据源地址（GitHub Actions服务器可直接访问）
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

# ========== 内部处理函数 ==========
def download_season_league(season, league_code):
    """下载单个联赛单个赛季的原始数据"""
    url = BASE_URL.format(season=season, league_code=league_code)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # 只读取需要的5个核心字段
        df = pd.read_csv(StringIO(resp.text), usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        return df
    except Exception as e:
        print(f"⚠️ {league_code} 赛季 {season} 下载失败: {str(e)}")
        return pd.DataFrame()

def process_data(df, league_std):
    """数据清洗、格式转换、队名统一、打联赛标签"""
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
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)
    
    # 打上联赛标识
    df["league"] = league_std
    
    # 数据类型校验与清洗
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["home_goals", "away_goals"])
    
    return df

# ========== 主执行流程 ==========
def main():
    all_leagues_data = []
    
    for data_code, std_code in LEAGUE_MAP.items():
        print(f"\n===== 处理联赛: {std_code}（数据源编码: {data_code}）=====")
        league_seasons_data = []
        
        for season in SEASONS:
            print(f"  正在下载赛季: {season}")
            raw_df = download_season_league(season, data_code)
            if raw_df.empty:
                continue
            processed_df = process_data(raw_df, std_code)
            league_seasons_data.append(processed_df)
            print(f"  ✅ 本赛季加载 {len(processed_df)} 场比赛")
        
        if league_seasons_data:
            league_all = pd.concat(league_seasons_data, ignore_index=True)
            all_leagues_data.append(league_all)
            print(f"  📊 {std_code} 总计 {len(league_all)} 场比赛")
        else:
            print(f"  ❌ {std_code} 未获取到任何数据")
    
    if not all_leagues_data:
        print("\n❌ 未获取到任何联赛数据，流程终止")
        return

    # 合并所有联赛所有赛季数据
    history_df = pd.concat(all_leagues_data, ignore_index=True)
    
    # 和现有数据合并、按比赛唯一标识去重
    data_path = "data/matches.csv"
    if os.path.exists(data_path):
        existing_df = pd.read_csv(data_path)
        # 兼容旧数据（没有league字段的默认标记为英超）
        if "league" not in existing_df.columns:
            existing_df["league"] = "PL"
            print("ℹ️ 检测到旧版数据，已默认标记为英超联赛")
        
        history_df = pd.concat([existing_df, history_df], ignore_index=True)
        # 按「日期+联赛+主队+客队」去重，保留最新版本
        history_df = history_df.drop_duplicates(
            subset=["date", "league", "home_team", "away_team"], 
            keep="last"
        )
    
    # 按日期升序排序（ELO模型依赖时间顺序）
    history_df = history_df.sort_values("date").reset_index(drop=True)
    
    # 确保目录存在并保存
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    history_df.to_csv(data_path, index=False, encoding="utf-8")
    
    print("\n" + "="*50)
    print(f"✅ 全部联赛历史数据初始化完成")
    print(f"总计比赛场数: {len(history_df)}")
    # 按联赛统计
    league_counts = history_df["league"].value_counts().to_dict()
    for league, count in sorted(league_counts.items()):
        print(f"  {league}: {count} 场")
    print(f"数据已保存至: {data_path}")
    print("="*50)

if __name__ == "__main__":
    main()

import pandas as pd
import requests
import os
from io import StringIO
from team_config import normalize_team_name

# ========== 配置项 ==========
# 赛季列表：2223=2022/23赛季，共4个完整赛季+当前赛季
SEASONS = ["2223", "2324", "2425", "2526"]

# 联赛映射：数据源编码 → 系统标准编码
# 数据源编码参考：E0=英超, SP1=西甲, D1=德甲, I1=意甲, F1=法甲
LEAGUE_MAP = {
    "E0": "PL",    # 英超
    "SP1": "PD",   # 西甲
    "D1": "BL1",   # 德甲
    "I1": "SA",    # 意甲
    "F1": "FL1"    # 法甲
}

# 历史数据数据源（稳定可用）
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

# ========== 内部函数 ==========
def download_season_league(season, league_code):
    """下载单个联赛单个赛季的原始数据（含半场比分）"""
    url = BASE_URL.format(season=season, league_code=league_code)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(
            StringIO(resp.text),
            usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]
        )
        return df
    except Exception as e:
        print(f"⚠️ {league_code} 赛季 {season} 下载失败: {str(e)}")
        return pd.DataFrame()

def process_data(df, league_std):
    """数据清洗、格式转换、队名统一、打联赛标签"""
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "HTHG": "ht_home_goals",
        "HTAG": "ht_away_goals"
    })

    # 日期格式统一（兼容多种日期格式）
    df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.strftime("%Y-%m-%d")

    # 统一队名
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    # 打上联赛标识
    df["league"] = league_std

    # 数据类型校验
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df["ht_home_goals"] = pd.to_numeric(df["ht_home_goals"], errors="coerce").astype("Int64")
    df["ht_away_goals"] = pd.to_numeric(df["ht_away_goals"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["home_goals", "away_goals"])

    return df

# ========== 主流程 ==========
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

    # 合并所有联赛所有赛季
    history_df = pd.concat(all_leagues_data, ignore_index=True)

    # 和现有数据合并 + 去重
    data_path = "data/matches.csv"
    if os.path.exists(data_path):
        existing_df = pd.read_csv(data_path)
        if "league" not in existing_df.columns:
            existing_df["league"] = "PL"
            print("ℹ️ 检测到旧版数据，已默认标记为英超联赛")

        # 旧数据可能没有半场列，补齐为空
        for col in ["ht_home_goals", "ht_away_goals"]:
            if col not in existing_df.columns:
                existing_df[col] = None

        history_df = pd.concat([existing_df, history_df], ignore_index=True)
        history_df = history_df.drop_duplicates(
            subset=["date", "league", "home_team", "away_team"],
            keep="last"
        )

    # 按日期升序（ELO模型强依赖时间顺序）
    history_df = history_df.sort_values("date").reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    history_df.to_csv(data_path, index=False, encoding="utf-8")

    # 输出统计
    print("\n" + "=" * 55)
    print(f"✅ 全部联赛历史数据初始化完成")
    print(f"总计比赛场数: {len(history_df)}")
    print("-" * 30)
    league_counts = history_df["league"].value_counts().sort_index().to_dict()
    for league, count in league_counts.items():
        print(f"  {league}: {count} 场")
    print("-" * 30)

    # 新增：半场数据统计
    if "ht_home_goals" in history_df.columns:
        has_ht = history_df["ht_home_goals"].notna().sum()
        missing_ht = history_df["ht_home_goals"].isna().sum()
        print(f"半场数据: 有 {has_ht} 场, 缺失 {missing_ht} 场")
        print("-" * 30)

        # 各联赛半场进球比
        for league in LEAGUE_MAP.values():
            lf = history_df[history_df["league"] == league].dropna(subset=["ht_home_goals", "ht_away_goals"])
            if len(lf) > 0:
                total = lf["home_goals"].sum() + lf["away_goals"].sum()
                ht_total = lf["ht_home_goals"].sum() + lf["ht_away_goals"].sum()
                ratio = ht_total / total if total > 0 else 0
                ht_draw = len(lf[lf["ht_home_goals"] == lf["ht_away_goals"]])
                ht_draw_rate = ht_draw / len(lf)
                print(f"  {league}: 半场进球比 {ratio:.3f}, 半场平局率 {ht_draw_rate:.1%} ({len(lf)}场)")
        print("-" * 30)

    print(f"数据已保存至: {data_path}")
    print("=" * 55)

if __name__ == "__main__":
    main()

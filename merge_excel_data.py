"""
历史比赛数据合并脚本
自动读取data文件夹下的Excel和CSV文件，清洗后合并到matches.csv
支持不同格式、不同列名、多Sheet自动识别
"""

import pandas as pd
import numpy as np
import os
import re

# ========== 配置 ==========
DATA_DIR = "data"
MATCHES_PATH = os.path.join(DATA_DIR, "matches.csv")

# 列名映射表：支持各种写法 → 标准列名
COLUMN_MAP = {
    # 日期
    "date": "date", "比赛日期": "date", "start_datetime": "date", "日期": "date",
    # 联赛
    "league": "league", "联赛": "league", "league_name": "league",
    # 主队
    "home_team": "home_team", "主队": "home_team", "home": "home_team",
    # 客队
    "away_team": "away_team", "客队": "away_team", "away": "away_team",
    # 全场进球
    "home_goals_ft": "home_goals", "home_goals": "home_goals", "主队进球": "home_goals",
    "away_goals_ft": "away_goals", "away_goals": "away_goals", "客队进球": "away_goals",
    # 半场进球
    "home_goals_1h": "ht_home_goals", "ht_home_goals": "ht_home_goals", "主队半场": "ht_home_goals",
    "away_goals_1h": "ht_away_goals", "ht_away_goals": "ht_away_goals", "客队半场": "ht_away_goals",
    # 赛季
    "season": "season", "赛季": "season",
}

# 联赛代码映射
LEAGUE_NAME_MAP = {
    "英超": "PL", "Premier League": "PL", "英格兰超级联赛": "PL",
    "西甲": "PD", "LaLiga": "PD", "La Liga": "PD", "西班牙甲级联赛": "PD",
    "德甲": "BL1", "Bundesliga": "BL1", "德国甲级联赛": "BL1",
    "意甲": "SA", "Serie A": "SA", "意大利甲级联赛": "SA",
    "法甲": "FL1", "Ligue 1": "FL1", "法国甲级联赛": "FL1",
}

def clean_team_name(name):
    """清洗球队名，统一格式"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r' FC$', '', name, flags=re.IGNORECASE)
    name = re.sub(r' CF$', '', name, flags=re.IGNORECASE)
    name = name.strip()
    return name

def parse_date(date_val):
    """解析日期，支持多种格式"""
    if pd.isna(date_val):
        return None
    if isinstance(date_val, pd.Timestamp):
        return date_val.strftime("%Y-%m-%d")
    date_str = str(date_val).strip()
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d %H:%M"]:
        try:
            return pd.to_datetime(date_str, format=fmt).strftime("%Y-%m-%d")
        except:
            continue
    try:
        return pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except:
        return None

def standardize_df(df):
    """标准化DataFrame列名和数据"""
    if len(df) == 0:
        return None
    
    # 列名小写、去空格
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 列名映射
    rename_dict = {}
    for col in df.columns:
        col_clean = col.strip().lower()
        if col_clean in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[col_clean]
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # 检查必要列
    required = ["home_team", "away_team"]
    if not all(c in df.columns for c in required):
        return None
    
    # 标准化联赛
    if "league" in df.columns:
        df["league"] = df["league"].astype(str).str.strip()
        df["league"] = df["league"].map(LEAGUE_NAME_MAP).fillna(df["league"])
    
    # 日期
    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_date)
        df = df.dropna(subset=["date"])
    
    # 进球字段转数字
    for col in ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 球队名清洗
    df["home_team"] = df["home_team"].apply(clean_team_name)
    df["away_team"] = df["away_team"].apply(clean_team_name)
    
    # 只保留需要的列
    keep_cols = ["date", "league", "home_team", "away_team", 
                "home_goals", "away_goals", 
                "ht_home_goals", "ht_away_goals", "season"]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].copy()
    
    df["status"] = "FINISHED"
    return df

def read_file(file_path):
    """读取单个文件，支持xlsx和csv"""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    
    print(f"\n📖 读取文件：{filename}")
    
    try:
        if ext in [".xlsx", ".xls"]:
            # Excel：读所有sheet
            xls = pd.ExcelFile(file_path)
            all_dfs = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"   Sheet：{sheet_name}，{len(df)} 行")
                std = standardize_df(df)
                if std is not None:
                    all_dfs.append(std)
            
            if not all_dfs:
                return None
            return pd.concat(all_dfs, ignore_index=True)
        
        elif ext == ".csv":
            # CSV
            df = pd.read_csv(file_path)
            print(f"   共 {len(df)} 行")
            return standardize_df(df)
        
        else:
            print(f"   ⚠️ 不支持的格式")
            return None
            
    except Exception as e:
        print(f"   ❌ 读取失败：{e}")
        return None

def main():
    print("=" * 60)
    print("📊 比赛数据合并工具（支持Excel+CSV）")
    print("=" * 60)
    
    # 查找所有支持的文件
    data_files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("~$"):
            continue
        if f.lower().endswith((".xlsx", ".xls", ".csv")):
            # 跳过已有的数据文件
            if f.lower() in ["matches.csv", "odds.csv", "predictions.csv", 
                            "history.csv", "backtest_results.csv"]:
                continue
            data_files.append(os.path.join(DATA_DIR, f))
    
    if not data_files:
        print("❌ data文件夹里没有找到可合并的比赛数据文件")
        return
    
    print(f"找到 {len(data_files)} 个数据文件\n")
    
    # 读取所有文件
    all_data = []
    for f in data_files:
        df = read_file(f)
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"   ✅ 有效")
    
    if not all_data:
        print("\n❌ 没有读取到有效数据")
        return
    
    # 合并所有
    new_matches = pd.concat(all_data, ignore_index=True)
    
    # 去重：日期+联赛+主队+客队
    before = len(new_matches)
    new_matches = new_matches.drop_duplicates(
        subset=["date", "league", "home_team", "away_team"], 
        keep="last"
    )
    new_matches = new_matches.reset_index(drop=True)
    print(f"\n📊 去重后：{len(new_matches)} 场（去除重复 {before - len(new_matches)} 场）")
    
    # 加载现有matches.csv
    if os.path.exists(MATCHES_PATH):
        existing = pd.read_csv(MATCHES_PATH)
        print(f"\n📂 现有 matches.csv：{len(existing)} 场")
        
        # 列对齐
        for col in new_matches.columns:
            if col not in existing.columns:
                existing[col] = np.nan
        
        # 合并
        combined = pd.concat([existing, new_matches], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "league", "home_team", "away_team"], 
            keep="last"
        )
        combined = combined.sort_values("date").reset_index(drop=True)
        
        added = len(combined) - len(existing)
        print(f"✅ 合并完成：共 {len(combined)} 场，新增 {added} 场")
        
    else:
        combined = new_matches.sort_values("date").reset_index(drop=True)
        print(f"✅ 新建 matches.csv：{len(combined)} 场")
    
    # 保存
    combined.to_csv(MATCHES_PATH, index=False)
    
    # 统计
    print("\n" + "-" * 60)
    print("📈 数据统计")
    print("-" * 60)
    
    if "league" in combined.columns:
        print("\n按联赛：")
        for lg in sorted(combined["league"].dropna().unique()):
            cnt = len(combined[combined["league"] == lg])
            ht_cnt = combined[combined["league"] == lg]["ht_home_goals"].notna().sum()
            print(f"  {lg}：{cnt} 场，半场数据 {ht_cnt} 场（{ht_cnt/cnt:.1%}）")
    
    if "season" in combined.columns:
        print("\n按赛季：")
        for s in sorted(combined["season"].dropna().unique()):
            print(f"  {s}：{len(combined[combined['season'] == s])} 场")
    
    # 半场覆盖率
    total = len(combined)
    ht_coverage = combined["ht_home_goals"].notna().sum()
    print(f"\n半场数据总覆盖率：{ht_coverage}/{total} = {ht_coverage/total:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print("=" * 60)
    print("\n下一步：")
    print("  运行 python backtest.py 重新回测")

if __name__ == "__main__":
    main()

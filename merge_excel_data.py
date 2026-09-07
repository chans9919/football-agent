import pandas as pd
from team_name_normalizer import normalize_team_name

def merge_excel_data(excel_path: str, output_csv: str = "data/matches.csv"):
    """合并Excel数据，统一队名标准"""
    df = pd.read_excel(excel_path)
    
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team_name)
    
    # 半场数据有效性标记
    df["has_half_time"] = df["HTHG"].notna() & df["HTAG"].notna()
    
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"数据合并完成，共 {len(df)} 场，已保存到 {output_csv}")

if __name__ == "__main__":
    merge_excel_data("data/raw_data.xlsx")

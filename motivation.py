from typing import Dict

def calculate_motivation(
    home_rank: int,
    away_rank: int,
    home_points: int,
    away_points: int,
    match_importance: float = 1.0
) -> Dict[str, float]:
    """
    战意系数计算（当前默认1.0，预留扩展接口）
    后续可根据赛程、保级/争冠形势调整幅度
    """
    base = 1.0
    
    # 暂不启用修正，返回基准值
    return {
        "home_motivation": round(base, 4),
        "away_motivation": round(base, 4)
    }

from team_name_normalizer import normalize_team_name

# 球队配置统一使用标准化队名
TEAM_CONFIG = {
    "Manchester City": {"tier": 1, "league": "英超"},
    "Liverpool": {"tier": 1, "league": "英超"},
    "Arsenal": {"tier": 1, "league": "英超"},
    "Bayern Munich": {"tier": 1, "league": "德甲"},
    "Real Madrid": {"tier": 1, "league": "西甲"},
    "Barcelona": {"tier": 1, "league": "西甲"},
}

def get_team_config(team_name: str) -> dict:
    normalized = normalize_team_name(team_name)
    return TEAM_CONFIG.get(normalized, {"tier": 3, "league": "default"})

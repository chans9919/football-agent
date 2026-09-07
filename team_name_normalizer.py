import re
from typing import Optional

# 常见别名映射
ALIAS_MAP = {
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "liverpool": "Liverpool",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "tottenham": "Tottenham",
    "spurs": "Tottenham",
    "bayern munich": "Bayern Munich",
    "bayern": "Bayern Munich",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "real madrid": "Real Madrid",
    "barcelona": "Barcelona",
    "barca": "Barcelona",
    "psg": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "juventus": "Juventus",
    "milan": "AC Milan",
    "inter": "Inter Milan",
}

def normalize_team_name(name: str) -> str:
    """统一队名标准：去后缀、规范空格、别名映射"""
    if not name:
        return ""
    
    # 去除前后空格
    name = name.strip()
    
    # 去除FC/CF/AFC后缀（不区分大小写）
    name = re.sub(r'\s+FC$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+CF$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+AFC$', '', name, flags=re.IGNORECASE)
    
    # 规范多空格为单空格
    name = re.sub(r'\s+', ' ', name)
    
    # 别名映射
    key = name.lower()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    
    # 首字母大写
    return name.title()

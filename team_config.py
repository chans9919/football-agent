# team_config.py
# 统一球队名称映射，所有脚本共用

TEAM_NAME_MAP = {
    # 英超
    "Liverpool": "Liverpool FC",
    "Ipswich": "Ipswich Town FC",
    "Man City": "Manchester City FC",
    "Manchester City": "Manchester City FC",
    "Man United": "Manchester United FC",
    "Manchester United": "Manchester United FC",
    "Arsenal": "Arsenal FC",
    "Chelsea": "Chelsea FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Spurs": "Tottenham Hotspur FC",
    "Aston Villa": "Aston Villa FC",
    "Newcastle": "Newcastle United FC",
    "Brighton": "Brighton & Hove Albion FC",
    "West Ham": "West Ham United FC",
    "Brentford": "Brentford FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Fulham": "Fulham FC",
    "Bournemouth": "AFC Bournemouth",
    "Wolves": "Wolverhampton Wanderers FC",
    "Sheffield Utd": "Sheffield United FC",
    "Burnley": "Burnley FC",
    "Luton": "Luton Town FC",
    "Leicester": "Leicester City FC",
    "Southampton": "Southampton FC",
    "Leeds": "Leeds United FC",

    # 西甲
    "Real Madrid": "Real Madrid CF",
    "Barcelona": "FC Barcelona",
    "Atletico Madrid": "Atlético Madrid",
    "Sevilla": "Sevilla FC",
    "Real Sociedad": "Real Sociedad",
    "Real Betis": "Real Betis",
    "Villarreal": "Villarreal CF",
    "Valencia": "Valencia CF",
    "Athletic Bilbao": "Athletic Club",
    "Osasuna": "CA Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Espanyol": "RCD Espanyol",
    "Getafe": "Getafe CF",
    "Cadiz": "Cádiz CF",
    "Almeria": "UD Almería",
    "Granada": "Granada CF",
    "Celta Vigo": "RC Celta de Vigo",
    "Mallorca": "RCD Mallorca",
    "Girona": "Girona FC",
    "Alaves": "Deportivo Alavés",
    "Las Palmas": "UD Las Palmas",
    "Leganes": "CD Leganés",

    # 德甲
    "Bayern Munich": "FC Bayern München",
    "Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Frankfurt": "Eintracht Frankfurt",
    "Wolfsburg": "VfL Wolfsburg",
    "Gladbach": "Borussia Mönchengladbach",
    "Mainz": "1. FSV Mainz 05",
    "Freiburg": "SC Freiburg",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "Union Berlin": "1. FC Union Berlin",
    "Stuttgart": "VfB Stuttgart",
    "Augsburg": "FC Augsburg",
    "Werder Bremen": "SV Werder Bremen",
    "Koln": "1. FC Köln",
    "Schalke": "FC Schalke 04",
    "Hertha Berlin": "Hertha BSC",
    "Bochum": "VfL Bochum",
    "Heidenheim": "1. FC Heidenheim",
    "Darmstadt": "Darmstadt 98",
    "Holstein Kiel": "Holstein Kiel",
    "St Pauli": "FC St. Pauli",

    # 意甲
    "Juventus": "Juventus FC",
    "AC Milan": "AC Milan",
    "Inter": "FC Internazionale Milano",
    "Inter Milan": "FC Internazionale Milano",
    "Napoli": "SSC Napoli",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Atalanta": "Atalanta BC",
    "Fiorentina": "ACF Fiorentina",
    "Torino": "Torino FC",
    "Bologna": "Bologna FC",
    "Udinese": "Udinese Calcio",
    "Sassuolo": "US Sassuolo Calcio",
    "Verona": "Hellas Verona FC",
    "Lecce": "US Lecce",
    "Cagliari": "Cagliari Calcio",
    "Empoli": "Empoli FC",
    "Genoa": "Genoa CFC",
    "Salernitana": "US Salernitana",
    "Frosinone": "Frosinone Calcio",
    "Monza": "AC Monza",
    "Parma": "Parma Calcio 1913",
    "Como": "Como 1907",
    "Venezia": "Venezia FC",

    # 法甲
    "PSG": "Paris Saint-Germain FC",
    "Paris SG": "Paris Saint-Germain FC",
    "Marseille": "Olympique de Marseille",
    "Lyon": "Olympique Lyonnais",
    "Monaco": "AS Monaco FC",
    "Lille": "LOSC Lille",
    "Rennes": "Stade Rennais FC",
    "Nice": "OGC Nice",
    "Strasbourg": "RC Strasbourg Alsace",
    "Nantes": "FC Nantes",
    "Montpellier": "Montpellier HSC",
    "Brest": "Stade Brestois 29",
    "Reims": "Stade de Reims",
    "Lorient": "FC Lorient",
    "Clermont": "Clermont Foot",
    "Toulouse": "Toulouse FC",
    "Auxerre": "AJ Auxerre",
    "Angers": "Angers SCO",
    "Metz": "FC Metz",
    "Lens": "RC Lens",
    "Le Havre": "Havre AC",
    "Saint-Etienne": "AS Saint-Étienne",
    "Caen": "Stade Malherbe Caen",
    "Bordeaux": "FC Girondins de Bordeaux",
    "Troyes": "ESTAC Troyes",
    "Dijon": "Dijon FCO",
}


def normalize_team_name(name):
    """将常见简称或带后缀的名称统一为标准名称"""
    if not name:
        return name
    
    # ========== 第一步：先做通用清洗（修复问题五） ==========
    name = name.strip()
    # 统一标点和空格
    name = name.replace(".", "").replace("-", " ").replace("  ", " ")
    
    # ========== 第二步：精确字典匹配 ==========
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    # ========== 第三步：大小写不敏感匹配 ==========
    name_lower = name.lower()
    for key, value in TEAM_NAME_MAP.items():
        if key.lower() == name_lower:
            return value
    
    # ========== 第四步：反向匹配（增加长度限制，修复问题二） ==========
    # 至少4个字符才做反向匹配，避免"FC"、"U"这类短字符串误匹配
    if len(name_lower) >= 4:
        for key, value in TEAM_NAME_MAP.items():
            if name_lower in value.lower():
                return value
    
    # 都匹配不到返回清洗后的原名
    return name

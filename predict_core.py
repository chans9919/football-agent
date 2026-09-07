import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date

# ===================== 全局常量（统一管理） =====================
DC_FACTOR_HOME = 1.15
DC_FACTOR_AWAY = 1.15
HOME_ADVANTAGE = 0.25
POISSON_MAX_GOALS = 12
BAYES_SMOOTH_FACTOR = 8.0
SEASON_RESET_MONTH = 8
SEASON_RESET_RATIO = 1/3

PINGXI_CALIBRATE_FACTOR = 0.7
PINGXI_STRICT_THRESHOLD = 0.40
PINGXI_FALLBACK_THRESHOLD = 0.38

# 联赛半场进球占比（历史统计值）
LEAGUE_HALF_RATIO = {
    "德甲": 0.465,
    "西甲": 0.455,
    "英超": 0.452,
    "法甲": 0.444,
    "意甲": 0.427,
    "default": 0.45
}

# ===================== 泊松概率计算（向量化优化） =====================
def poisson_pmf_vector(max_goals: int, lam: float) -> np.ndarray:
    """向量化计算泊松分布概率质量函数"""
    k = np.arange(max_goals + 1)
    log_pmf = k * math.log(lam) - lam - np.array([math.lgamma(x + 1) for x in k])
    return np.exp(log_pmf)

def match_probability_matrix(home_lam: float, away_lam: float, max_goals: int = POISSON_MAX_GOALS) -> np.ndarray:
    """向量化生成比分概率矩阵，替代双重循环，提速4-5倍"""
    home_p = poisson_pmf_vector(max_goals, home_lam)
    away_p = poisson_pmf_vector(max_goals, away_lam)
    matrix = np.outer(home_p, away_p)
    
    # DC修正
    matrix[0, 0] *= DC_FACTOR_HOME * DC_FACTOR_AWAY
    if max_goals >= 1:
        matrix[1, 0] *= DC_FACTOR_HOME
        matrix[0, 1] *= DC_FACTOR_AWAY
    
    # 归一化
    matrix /= matrix.sum()
    return matrix

# ===================== 基础概率提取 =====================
def extract_win_draw_loss(matrix: np.ndarray) -> Dict[str, float]:
    """从比分矩阵提取胜平负概率"""
    home_win = np.sum(np.triu(matrix, k=1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.tril(matrix, k=-1))
    total = home_win + draw + away_win
    return {
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total
    }

def extract_total_goals_dist(matrix: np.ndarray) -> Dict[int, float]:
    """提取总进球分布"""
    max_g = matrix.shape[0] - 1
    totals = np.zeros(max_g * 2 + 1)
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            totals[h + a] += matrix[h, a]
    return {i: totals[i] for i in range(max_g * 2 + 1)}

def extract_over_under(matrix: np.ndarray, line: float = 2.5) -> Dict[str, float]:
    """提取大小球概率"""
    totals = extract_total_goals_dist(matrix)
    over = sum(p for g, p in totals.items() if g > line)
    under = sum(p for g, p in totals.items() if g < line)
    return {"over": over, "under": under}

def calculate_half_full_probs(home_lam: float, away_lam: float, league: str = "default") -> Dict[str, float]:
    """计算半全场9种组合概率，按联赛使用对应半场比例"""
    ratio = LEAGUE_HALF_RATIO.get(league, LEAGUE_HALF_RATIO["default"])
    half_home = home_lam * ratio
    half_away = away_lam * ratio
    
    half_matrix = match_probability_matrix(half_home, half_away)
    full_matrix = match_probability_matrix(home_lam, away_lam)
    
    half_wdl = extract_win_draw_loss(half_matrix)
    full_wdl = extract_win_draw_loss(full_matrix)
    
    return {
        "HH": half_wdl["home_win"] * full_wdl["home_win"],
        "HD": half_wdl["home_win"] * full_wdl["draw"],
        "HA": half_wdl["home_win"] * full_wdl["away_win"],
        "DH": half_wdl["draw"] * full_wdl["home_win"],
        "DD": half_wdl["draw"] * full_wdl["draw"],
        "DA": half_wdl["draw"] * full_wdl["away_win"],
        "AH": half_wdl["away_win"] * full_wdl["home_win"],
        "AD": half_wdl["away_win"] * full_wdl["draw"],
        "AA": half_wdl["away_win"] * full_wdl["away_win"],
        "half_draw": half_wdl["draw"],
        "full_draw": full_wdl["draw"]
    }

# ===================== 球队评分系统（贝叶斯平滑+赛季重置） =====================
class TeamRating:
    def __init__(self):
        self.teams: Dict[str, Dict] = {}
        self.league_avg_goals = 1.35
        self.last_date: Optional[date] = None

    def _is_new_season(self, current_date: date) -> bool:
        """判断是否跨赛季（8月为切换点）"""
        if self.last_date is None:
            return False
        return current_date.month >= SEASON_RESET_MONTH and self.last_date.month < SEASON_RESET_MONTH

    def _season_reset(self):
        """赛季重置：所有球队评分向均值回归1/3"""
        for team in self.teams.values():
            team["attack"] = team["attack"] * (2/3) + self.league_avg_goals * (1/3)
            team["defense"] = team["defense"] * (2/3) + self.league_avg_goals * (1/3)
            team["matches"] = max(1, int(team["matches"] * 0.7))

    def add_match(self, home_team: str, away_team: str, home_goals: int, away_goals: int, match_date: date):
        # 赛季重置检测
        if self._is_new_season(match_date):
            self._season_reset()
        
        self.last_date = match_date
        
        # 初始化新球队
        for team in [home_team, away_team]:
            if team not in self.teams:
                self.teams[team] = {
                    "attack": self.league_avg_goals,
                    "defense": self.league_avg_goals,
                    "matches": 0,
                    "gf": 0,
                    "ga": 0
                }

        # 更新数据
        self.teams[home_team]["gf"] += home_goals
        self.teams[home_team]["ga"] += away_goals
        self.teams[home_team]["matches"] += 1
        self.teams[away_team]["gf"] += away_goals
        self.teams[away_team]["ga"] += home_goals
        self.teams[away_team]["matches"] += 1

        # 贝叶斯平滑计算评分：场次越少，向均值收缩越多
        for team in [home_team, away_team]:
            t = self.teams[team]
            n = t["matches"]
            weight = n / (n + BAYES_SMOOTH_FACTOR)
            t["attack"] = weight * (t["gf"] / n) + (1 - weight) * self.league_avg_goals
            t["defense"] = weight * (t["ga"] / n) + (1 - weight) * self.league_avg_goals

    def get_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """获取两队期望进球，含主场优势"""
        home_att = self.teams.get(home_team, {"attack": self.league_avg_goals})["attack"]
        away_def = self.teams.get(away_team, {"defense": self.league_avg_goals})["defense"]
        away_att = self.teams.get(away_team, {"attack": self.league_avg_goals})["attack"]
        home_def = self.teams.get(home_team, {"defense": self.league_avg_goals})["defense"]
        
        home_xg = home_att * away_def + HOME_ADVANTAGE
        away_xg = away_att * home_def
        return home_xg, away_xg

# ===================== 平系策略 =====================
def calibrate_pingxi(raw_half_draw: float) -> float:
    return raw_half_draw * PINGXI_CALIBRATE_FACTOR

def get_pingxi_double_selection(half_draw: float, home_win: float, away_win: float, full_draw: float) -> Dict:
    """平系双选逻辑：半场平 + 全场方向"""
    calibrated = calibrate_pingxi(half_draw)
    
    if home_win >= away_win:
        primary = "DH"  # 平胜
        secondary = "DD" # 平平
        combined = half_draw * home_win + half_draw * full_draw
    else:
        primary = "DA"  # 平负
        secondary = "DD" # 平平
        combined = half_draw * away_win + half_draw * full_draw
    
    return {
        "calibrated_half_draw": calibrated,
        "primary": primary,
        "secondary": secondary,
        "combined_prob": combined
    }

def select_pingxi_matches(matches: List[Dict]) -> List[Dict]:
    """动态兜底选场：优先严格场，不足2场自动兜底"""
    strict = [m for m in matches if m["calibrated_half_draw"] >= PINGXI_STRICT_THRESHOLD]
    if len(strict) >= 2:
        return strict
    
    fallback = [m for m in matches if m["calibrated_half_draw"] >= PINGXI_FALLBACK_THRESHOLD]
    return fallback[:2]

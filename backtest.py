import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import poisson
from train_model import train_poisson, calculate_elo
from team_config import normalize_team_name

# ========== 配置 ==========
LEAGUES = ["PL", "PD", "BL1", "SA", "FL1"]
LEAGUE_NAMES = {"PL": "英超", "PD": "西甲", "BL1": "德甲", "SA": "意甲", "FL1": "法甲"}

LEAGUE_PARAMS = {
    "PL": {"draw_corr": 0.98}, "PD": {"draw_corr": 1.05},
    "BL1": {"draw_corr": 0.95}, "SA": {"draw_corr": 1.03},
    "FL1": {"draw_corr": 1.02},
}

LEAGUE_WEIGHTS = {
    "PL": [0.35, 0.65], "PD": [0.65, 0.35], "BL1": [0.35, 0.65],
    "SA": [0.45, 0.55], "FL1": [0.50, 0.50],
}

PING_STRICT = 0.48
PING_BALANCED = 0.44
MIN_TRAIN_MATCHES = 30
ELO_REBUILD_INTERVAL = 20  # 每20场重建ELO（原50场，更精确）

# 回测窗口：None=用全部历史数据，数字=最近N场
BACKTEST_WINDOW = None

# ========== 核心计算函数 ==========
def poisson_matrix(lh, la, max_g=8):
    m = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            m[i,j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    for i, j in [(0,0),(1,1),(2,2)]:
        if i < m.shape[0] and j < m.shape[1]:
            m[i,j] *= 1.30
    for i, j in [(1,0),(0,1)]:
        if i < m.shape[0] and j < m.shape[1]:
            m[i,j] *= 1.15
    m /= m.sum()
    return m

def elo_prob(he, ae, adv=65):
    diff = he + adv - ae
    dp = max(0.20, min(0.32, 0.30 - 0.07 * abs(diff)/400))
    ew = 1 / (1 + 10**(-diff/400))
    rem = 1 - dp
    hw, aw = ew * rem, (1-ew) * rem
    t = hw + dp + aw
    return hw/t, dp/t, aw/t

def odds_to_prob(oh, od, oa):
    """赔率转概率（归一化，去除抽水）"""
    if pd.isna(oh) or pd.isna(od) or pd.isna(oa) or oh <= 0 or od <= 0 or oa <= 0:
        return None
    ph, pd_, pa = 1/oh, 1/od, 1/oa
    t = ph + pd_ + pa
    return [ph/t, pd_/t, pa/t]

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def inv_logit(x):
    return 1/(1+np.exp(-x))

def fuse(probs_list, weights):
    logits = [logit(p) for p in probs_list]
    fl = sum(w*l for w,l in zip(weights, logits))
    return inv_logit(fl)

def calc_handicap(matrix, goals, fav_home=True):
    w, d, l = 0.0, 0.0, 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if fav_home:
                adj, opp = i - goals, j
            else:
                adj, opp = j - goals, i
            if goals in [0.5, 1.5]:
                if adj > opp: w += matrix[i,j]
                else: l += matrix[i,j]
            else:
                if adj > opp: w += matrix[i,j]
                elif adj == opp: d += matrix[i,j]
                else: l += matrix[i,j]
    return w, d, l

def predict_match(lh, la, half_ratio=0.44):
    matrix = poisson_matrix(lh, la)
    hw = np.sum(np.tril(matrix, -1))
    dr = np.sum(np.diag(matrix))
    aw = np.sum(np.triu(matrix, 1))
    
    # 比分概率TOP5
    score_probs = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            score_probs[f"{i}-{j}"] = matrix[i,j]
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 总进球概率
    total_goals = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            tg = i + j
            total_goals[tg] = total_goals.get(tg, 0) + matrix[i,j]
    top_tg = sorted(total_goals.items(), key=lambda x: x[1], reverse=True)[:5]
    over25 = sum(p for tg, p in total_goals.items() if tg >= 3)
    
    # 半场
    lhh, lah = lh * half_ratio, la * half_ratio
    hm = poisson_matrix(lhh, lah, max_g=4)
    hth = np.sum(np.tril(hm, -1))
    htd = np.sum(np.diag(hm))
    hta = np.sum(np.triu(hm, 1))
    
    htft = {
        "胜胜": hth*hw, "胜平": hth*dr, "胜负": hth*aw,
        "平胜": htd*hw, "平平": htd*dr, "平负": htd*aw,
        "负胜": hta*hw, "负平": hta*dr, "负负": hta*aw,
    }
    top_htft = sorted(htft.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # 让球
    fav_home = lh >= la
    levels = [0.5, 1.0, 1.5, 2.0]
    hc_all = {lv: calc_handicap(matrix, lv, fav_home) for lv in levels}
    if hc_all[0.5][0] < 0.40:
        handicap = {"is_draw": True, "win": 0, "draw": 0, "lose": 0, "goals": None, "fav": "home" if fav_home else "away"}
    else:
        best = min(levels, key=lambda lv: abs(hc_all[lv][0] - 0.5))
        w, d, l = hc_all[best]
        handicap = {"is_draw": False, "win": w, "draw": d, "lose": l, "goals": best, "fav": "home" if fav_home else "away"}
    
    return {
        "poisson": [hw, dr, aw],
        "ht_draw": htd,
        "htft": htft,
        "top_htft": top_htft,
        "handicap": handicap,
        "top_scores": top_scores,
        "top_tg": top_tg,
        "over25": over25,
    }

def grade(probs):
    mx = max(probs)
    idx = np.argmax(probs)
    if mx >= 0.5: g = "A档"
    elif mx >= 0.4: g = "B档"
    elif mx >= 0.35: g = "C档"
    else: g = "不推荐"
    return g, idx, mx

def actual_result(row):
    hg, ag = row["home_goals"], row["away_goals"]
    if hg > ag: return 0
    elif hg == ag: return 1
    else: return 2

def actual_htft(row):
    hg, ag = row["home_goals"], row["away_goals"]
    hthg = row.get("ht_home_goals")
    htag = row.get("ht_away_goals")
    if pd.isna(hthg) or pd.isna(htag):
        return None
    ht = "胜" if hthg > htag else ("平" if hthg == htag else "负")
    ft = "胜" if hg > ag else ("平" if hg == ag else "负")
    return ht + ft

# ========== 主回测函数 ==========
def run_backtest():
    if not os.path.exists("data/matches.csv"):
        print("❌ 找不到 data/matches.csv")
        return
    
    df = pd.read_csv("data/matches.csv")
    print(f"📚 总历史比赛：{len(df)} 场")
    
    # 清洗
    if "status" in df.columns:
        df = df[df["status"].isin(["FINISHED", "finished", "已完赛"])]
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"✅ 有效已完赛：{len(df)} 场")
    
    # 尝试加载历史赔率
    odds_df = None
    has_odds = False
    if os.path.exists("data/odds.csv"):
        try:
            odds_df = pd.read_csv("data/odds.csv")
            odds_df["date"] = pd.to_datetime(odds_df["date"], errors="coerce")
            has_odds = True
            print(f"📊 检测到历史赔率数据：{len(odds_df)} 条")
        except Exception as e:
            print(f"⚠️ 赔率数据加载失败：{e}，回测不含市场赔率")
    
    # 回测窗口
    if BACKTEST_WINDOW and len(df) > BACKTEST_WINDOW:
        df = df.iloc[-BACKTEST_WINDOW:].reset_index(drop=True)
        print(f"📊 回测窗口：最近 {len(df)} 场\n")
    else:
        print(f"📊 回测窗口：全部 {len(df)} 场历史数据\n")
    
    results = []
    elo_cache = {}
    elo_rebuild_counter = {}
    
    print(f"🔄 开始滑动窗口回测（每场只用之前数据训练）...\n")
    
    for idx in range(len(df)):
        if idx < MIN_TRAIN_MATCHES:
            continue
        
        row = df.iloc[idx]
        league = row.get("league", "PL")
        if league not in LEAGUES:
            continue
        
        home_en = normalize_team_name(row["home_team"])
        away_en = normalize_team_name(row["away_team"])
        
        train_df = df.iloc[:idx]
        if "league" in train_df.columns:
            lg_train = train_df[train_df["league"] == league]
        else:
            lg_train = train_df
        
        if len(lg_train) < 10:
            continue
        
        try:
            lh, la = train_poisson(lg_train, home_en, away_en)
            pred = predict_match(lh, la)
            
            # 分联赛校准
            params = LEAGUE_PARAMS.get(league, LEAGUE_PARAMS["PL"])
            p_poi_raw = pred["poisson"]
            p_poi = [p_poi_raw[0], p_poi_raw[1] * params["draw_corr"], p_poi_raw[2]]
            tp = sum(p_poi)
            p_poi = [p/tp for p in p_poi]
            
            # ELO（每20场重建）
            if league not in elo_cache or idx - elo_rebuild_counter.get(league, 0) >= ELO_REBUILD_INTERVAL:
                elo_dict = calculate_elo(lg_train)
                elo_cache[league] = elo_dict
                elo_rebuild_counter[league] = idx
            else:
                elo_dict = elo_cache[league]
            
            he = elo_dict.get(home_en, 1500)
            ae = elo_dict.get(away_en, 1500)
            p_elo = list(elo_prob(he, ae))
            
            # 市场赔率（如果有）
            p_market = None
            if has_odds and odds_df is not None:
                match_odds = odds_df[
                    (odds_df["date"] == row["date"]) &
                    (odds_df["home_team"] == row["home_team"]) &
                    (odds_df["away_team"] == row["away_team"])
                ]
                if len(match_odds) > 0:
                    mo = match_odds.iloc[0]
                    p_market = odds_to_prob(mo.get("odds_home"), mo.get("odds_draw"), mo.get("odds_away"))
            
            # 融合（泊松+ELO）
            weights = LEAGUE_WEIGHTS.get(league, [0.5, 0.5])
            tw = sum(weights)
            wn = [w/tw for w in weights]
            fh = fuse([p_poi[0], p_elo[0]], wn)
            fd = fuse([p_poi[1], p_elo[1]], wn)
            fa = fuse([p_poi[2], p_elo[2]], wn)
            tf = fh + fd + fa
            fused = [fh/tf, fd/tf, fa/tf]
            
            # 三模型融合（含市场赔率）
            fused_full = None
            if p_market is not None:
                w3 = [0.3, 0.3, 0.4]  # 泊松30% + ELO30% + 市场40%
                fh3 = fuse([p_poi[0], p_elo[0], p_market[0]], w3)
                fd3 = fuse([p_poi[1], p_elo[1], p_market[1]], w3)
                fa3 = fuse([p_poi[2], p_elo[2], p_market[2]], w3)
                tf3 = fh3 + fd3 + fa3
                fused_full = [fh3/tf3, fd3/tf3, fa3/tf3]
            
            grade_name, dir_idx, max_p = grade(fused)
            actual = actual_result(row)
            act_htft = actual_htft(row)
            
            # 实际比分和总进球
            actual_score = f"{int(row['home_goals'])}-{int(row['away_goals'])}"
            actual_tg = int(row["home_goals"] + row["away_goals"])
            
            # 比分TOP3命中
            top3_scores = [s[0] for s in pred["top_scores"][:3]]
            score_top3_hit = actual_score in top3_scores
            score_top1_hit = pred["top_scores"][0][0] == actual_score
            
            # 总进球TOP3命中
            top3_tg = [t[0] for t in pred["top_tg"][:3]]
            tg_top3_hit = actual_tg in top3_tg
            tg_top1_hit = pred["top_tg"][0][0] == actual_tg
            
            # 大小球命中
            actual_over25 = actual_tg >= 3
            over25_hit = (pred["over25"] >= 0.5) == actual_over25
            
            # ===== 让球盘（修复走盘处理）=====
            hc = pred["handicap"]
            hc_hit = None
            hc_draw = False  # 走盘标记
            if not hc["is_draw"]:
                hg, ag = row["home_goals"], row["away_goals"]
                if hc["fav"] == "home":
                    adj, opp = hg - hc["goals"], ag
                else:
                    adj, opp = ag - hc["goals"], hg
                
                if hc["goals"] in [0.5, 1.5]:
                    # 半球/球半：无走盘
                    hc_hit = adj > opp
                else:
                    # 整数球：有走盘
                    if adj > opp:
                        hc_hit = True
                    elif adj == opp:
                        hc_draw = True  # 走盘，不算赢也不算输
                        hc_hit = None
                    else:
                        hc_hit = False
            
            # 平系
            htd = pred["ht_draw"]
            if htd >= PING_STRICT:
                ping_type = "严格平系"
            elif htd >= PING_BALANCED:
                ping_type = "均衡平系"
            else:
                ping_type = "非平系"
            
            # ===== 半全场多组合命中率统计 =====
            hf_combo_hits = {}
            if act_htft:
                # 各种双选组合
                hf_combo_hits["平平+平胜"] = act_htft in ["平平", "平胜"]
                hf_combo_hits["平平+平负"] = act_htft in ["平平", "平负"]
                hf_combo_hits["平平+胜平"] = act_htft in ["平平", "胜平"]
                hf_combo_hits["平平+负平"] = act_htft in ["平平", "负平"]
                hf_combo_hits["胜胜+平胜"] = act_htft in ["胜胜", "平胜"]
                hf_combo_hits["负负+平负"] = act_htft in ["负负", "平负"]
                hf_combo_hits["胜胜+平平"] = act_htft in ["胜胜", "平平"]
                hf_combo_hits["负负+平平"] = act_htft in ["负负", "平平"]
            
            # 平系双选（按全场方向对齐）
            hf_hit = None
            if act_htft and ping_type in ["严格平系", "均衡平系"]:
                if dir_idx == 0:
                    hf_hit = act_htft in ["平平", "平胜"]
                elif dir_idx == 2:
                    hf_hit = act_htft in ["平平", "平负"]
                else:
                    hf_hit = act_htft in ["平平", "平胜"]
            
            # 半全场TOP3命中
            htft_top3_hit = None
            if act_htft:
                top3_htft = [h[0] for h in pred["top_htft"][:3]]
                htft_top3_hit = act_htft in top3_htft
            
            results.append({
                "date": row["date"], "league": league,
                "home": home_en, "away": away_en,
                "fused": fused, "fused_full": fused_full,
                "grade": grade_name,
                "dir_idx": dir_idx, "max_prob": max_p,
                "actual": actual, "hit": dir_idx == actual,
                "ht_draw": htd, "ping_type": ping_type,
                "act_htft": act_htft, "hf_hit": hf_hit,
                "hf_combo_hits": hf_combo_hits,
                "htft_top3_hit": htft_top3_hit,
                "handicap": hc, "hc_hit": hc_hit, "hc_draw": hc_draw,
                "poisson": p_poi, "elo": p_elo, "market": p_market,
                "actual_score": actual_score, "actual_tg": actual_tg,
                "score_top1_hit": score_top1_hit, "score_top3_hit": score_top3_hit,
                "tg_top1_hit": tg_top1_hit, "tg_top3_hit": tg_top3_hit,
                "over25_hit": over25_hit, "pred_over25": pred["over25"],
            })
            
        except Exception as e:
            continue
        
        if (idx + 1) % 200 == 0:
            print(f"  进度：{idx+1}/{len(df)}，有效回测 {len(results)} 场")
    
    print(f"\n✅ 回测完成，有效样本 {len(results)} 场")
    
    if not results:
        print("❌ 无有效回测结果")
        return
    
    # ========== 生成回测报告 ==========
    report = "# 模型全面历史回测报告\n\n"
    report += f"**回测时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**回测窗口**：{len(df)} 场历史比赛，有效预测 {len(results)} 场\n"
    report += f"**模型**：泊松（DC修正+联赛校准）+ ELO 融合"
    if has_odds:
        report += " + 市场赔率（三模型融合对比）"
    report += "\n"
    report += f"**回测方式**：滑动窗口，每场比赛只用该场之前的数据训练\n"
    report += f"**ELO更新频率**：每{ELO_REBUILD_INTERVAL}场重建一次\n\n"
    report += "> ⚠️ 说明：基础融合（泊松+ELO）不含市场赔率；三模型融合含市场赔率。回测结果仅供模型校准参考，不构成投注建议。\n\n"
    report += "---\n\n"
    
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    
    # ===== 一、胜平负 =====
    report += "## 一、胜平负预测准确率\n\n"
    report += f"**总体命中率（泊松+ELO融合）**：{hits}/{total} = **{hits/total:.1%}**\n\n"
    
    # 三模型融合对比
    full_results = [r for r in results if r["fused_full"] is not None]
    if full_results:
        full_hits = sum(1 for r in full_results if np.argmax(r["fused_full"]) == r["actual"])
        base_hits_sub = sum(1 for r in full_results if r["hit"])
        report += f"**含市场赔率三模型融合命中率**：{full_hits}/{len(full_results)} = **{full_hits/len(full_results):.1%}**"
        report += f"（同期基础融合 {base_hits_sub/len(full_results):.1%}，"
        diff = full_hits/len(full_results) - base_hits_sub/len(full_results)
        report += f"市场赔率{'提升' if diff >= 0 else '降低'} {abs(diff):.1%}）\n\n"
    
    report += "### 1.1 按档位统计\n\n"
    report += "| 档位 | 场次 | 命中 | 命中率 | 平均预测概率 | 概率偏差 |\n"
    report += "|------|------|------|--------|-------------|----------|\n"
    for g in ["A档", "B档", "C档", "不推荐"]:
        gr = [r for r in results if r["grade"] == g]
        if not gr:
            continue
        gh = sum(1 for r in gr if r["hit"])
        ap = np.mean([r["max_prob"] for r in gr])
        ar = gh / len(gr)
        report += f"| {g} | {len(gr)} | {gh} | {ar:.1%} | {ap:.1%} | {ar-ap:+.1%} |\n"
    report += "\n> 概率偏差 = 实际命中率 - 平均预测概率。正数=模型低估，负数=模型高估。\n\n"
    
    report += "### 1.2 按联赛统计\n\n"
    report += "| 联赛 | 场次 | 命中率 |\n|------|------|--------|\n"
    for lg in LEAGUES:
        lr = [r for r in results if r["league"] == lg]
        if not lr:
            continue
        lh = sum(1 for r in lr if r["hit"])
        report += f"| {LEAGUE_NAMES.get(lg, lg)} | {len(lr)} | {lh/len(lr):.1%} |\n"
    report += "\n---\n\n"
    
    # ===== 二、概率校准 + 模型对比 =====
    report += "## 二、概率校准与模型对比\n\n"
    report += "### 2.1 概率校准分析\n\n"
    report += "| 预测概率区间 | 场次 | 实际命中率 | 偏差 |\n|--------------|------|-----------|------|\n"
    bins = [(0.60,1.0,"≥60%"), (0.50,0.60,"50%~60%"), (0.45,0.50,"45%~50%"),
            (0.40,0.45,"40%~45%"), (0.35,0.40,"35%~40%"), (0,0.35,"<35%")]
    for low, high, label in bins:
        br = [r for r in results if low <= r["max_prob"] < high]
        if not br:
            continue
        bh = sum(1 for r in br if r["hit"])
        ap = np.mean([r["max_prob"] for r in br])
        ar = bh / len(br)
        report += f"| {label} | {len(br)} | {ar:.1%} | {ar-ap:+.1%} |\n"
    report += "\n"
    
    report += "### 2.2 各模型对比\n\n"
    report += "| 模型 | 命中率 |\n|------|--------|\n"
    for model_name, key in [("泊松模型", "poisson"), ("ELO模型", "elo"), ("融合模型（泊松+ELO）", "fused")]:
        mh = sum(1 for r in results if np.argmax(r[key]) == r["actual"])
        report += f"| {model_name} | {mh/total:.1%} |\n"
    if full_results:
        mh_full = sum(1 for r in full_results if np.argmax(r["fused_full"]) == r["actual"])
        report += f"| 三模型融合（含市场赔率） | {mh_full/len(full_results):.1%}（{len(full_results)}场有赔率） |\n"
    report += "\n---\n\n"
    
    # ===== 三、比分预测 =====
    report += "## 三、比分预测准确率\n\n"
    s1 = sum(1 for r in results if r["score_top1_hit"])
    s3 = sum(1 for r in results if r["score_top3_hit"])
    report += f"- **TOP1比分命中率**：{s1}/{total} = **{s1/total:.1%}**\n"
    report += f"- **TOP3比分命中率**：{s3}/{total} = **{s3/total:.1%}**\n\n"
    
    report += "### 3.1 按实际总进球统计TOP3命中率\n\n"
    report += "| 实际总进球 | 场次 | TOP3命中 | 命中率 |\n|-----------|------|----------|--------|\n"
    tg_bins = [(0,1,"0-1球"), (2,2,"2球"), (3,3,"3球"), (4,4,"4球"), (5,99,"5球+")]
    for low, high, label in tg_bins:
        br = [r for r in results if low <= r["actual_tg"] <= high]
        if not br:
            continue
        bh = sum(1 for r in br if r["score_top3_hit"])
        report += f"| {label} | {len(br)} | {bh} | {bh/len(br):.1%} |\n"
    report += "\n---\n\n"
    
    # ===== 四、总进球数预测 =====
    report += "## 四、总进球数预测准确率\n\n"
    t1 = sum(1 for r in results if r["tg_top1_hit"])
    t3 = sum(1 for r in results if r["tg_top3_hit"])
    o25 = sum(1 for r in results if r["over25_hit"])
    report += f"- **TOP1总进球命中率**：{t1}/{total} = **{t1/total:.1%}**\n"
    report += f"- **TOP3总进球命中率**：{t3}/{total} = **{t3/total:.1%}**\n"
    report += f"- **大小球（2.5球）准确率**：{o25}/{total} = **{o25/total:.1%}**\n\n"
    
    report += "### 4.1 按实际总进球统计\n\n"
    report += "| 实际总进球 | 场次 | 占比 | TOP1命中 | TOP3命中 |\n|-----------|------|------|----------|----------|\n"
    for tg_val in range(0, 8):
        br = [r for r in results if r["actual_tg"] == tg_val]
        if not br:
            continue
        bh1 = sum(1 for r in br if r["tg_top1_hit"])
        bh3 = sum(1 for r in br if r["tg_top3_hit"])
        report += f"| {tg_val}球 | {len(br)} | {len(br)/total:.1%} | {bh1/len(br):.1%} | {bh3/len(br):.1%} |\n"
    report += "\n"
    
    report += "### 4.2 大小球校准\n\n"
    report += "| 预测大球概率区间 | 场次 | 实际大球率 | 偏差 |\n|-----------------|------|-----------|------|\n"
    o_bins = [(0.70,1.0,"≥70%"), (0.60,0.70,"60%~70%"), (0.50,0.60,"50%~60%"),
               (0.40,0.50,"40%~50%"), (0,0.40,"<40%")]
    for low, high, label in o_bins:
        br = [r for r in results if low <= r["pred_over25"] < high]
        if not br:
            continue
        actual_o = sum(1 for r in br if r["actual_tg"] >= 3) / len(br)
        pred_o = np.mean([r["pred_over25"] for r in br])
        report += f"| {label} | {len(br)} | {actual_o:.1%} | {actual_o-pred_o:+.1%} |\n"
    report += "\n---\n\n"
    
    # ===== 五、让球盘（修复走盘后统计）=====
    report += "## 五、让球盘预测准确率\n\n"
    report += "> 📌 统计说明：整数让球（1球/2球）走盘不计入赢盘率分母（本金退还，相当于未投注）。\n\n"
    
    hc_all_results = [r for r in results if not r["handicap"]["is_draw"]]
    hc_settled = [r for r in hc_all_results if r["hc_hit"] is not None]  # 排除走盘
    hc_draw_count = sum(1 for r in hc_all_results if r["hc_draw"])
    
    if hc_settled:
        hch = sum(1 for r in hc_settled if r["hc_hit"])
        report += f"**让球盘总体赢盘率**（排除走盘）：{hch}/{len(hc_settled)} = **{hch/len(hc_settled):.1%}**\n"
        report += f"- 走盘场次：{hc_draw_count} 场（占有让球盘比赛的 {hc_draw_count/len(hc_all_results):.1%}）\n"
        report += f"- 平手盘（无让球参考）：{sum(1 for r in results if r['handicap']['is_draw'])} 场\n\n"
        
        report += "### 5.1 按预测赢盘概率分档\n\n"
        report += "| 预测赢盘概率 | 场次（已结算） | 走盘 | 实际赢盘率 | 偏差 |\n|-------------|----------------|------|-----------|------|\n"
        hc_bins = [(0.50,1.0,"≥50%（强稳胆）"), (0.45,0.50,"45%~50%（稳胆）"),
                    (0.40,0.45,"40%~45%（准稳胆）"), (0,0.40,"<40%（价值低）")]
        for low, high, label in hc_bins:
            br_all = [r for r in hc_all_results if low <= r["handicap"]["win"] < high]
            br_settled = [r for r in br_all if r["hc_hit"] is not None]
            br_draw = sum(1 for r in br_all if r["hc_draw"])
            if not br_settled:
                continue
            bh = sum(1 for r in br_settled if r["hc_hit"])
            ap = np.mean([r["handicap"]["win"] for r in br_settled])
            ar = bh / len(br_settled)
            report += f"| {label} | {len(br_settled)} | {br_draw} | {ar:.1%} | {ar-ap:+.1%} |\n"
        report += "\n"
        
        report += "### 5.2 按让球档位统计\n\n"
        report += "| 让球档位 | 场次 | 赢盘 | 走盘 | 输盘 | 赢盘率（排除走盘）|\n|---------|------|------|------|------|------------------|\n"
        for goals in [0.5, 1.0, 1.5, 2.0]:
            br = [r for r in hc_all_results if r["handicap"]["goals"] == goals]
            if not br:
                continue
            bw = sum(1 for r in br if r["hc_hit"] == True)
            bd = sum(1 for r in br if r["hc_draw"])
            bl = sum(1 for r in br if r["hc_hit"] == False)
            settled = bw + bl
            wr = bw / settled if settled > 0 else 0
            report += f"| 让{goals}球 | {len(br)} | {bw} | {bd} | {bl} | {wr:.1%} |\n"
        report += "\n"
    else:
        report += "无让球盘回测数据。\n\n"
    report += "---\n\n"
    
    # ===== 六、半全场 =====
    report += "## 六、半全场预测准确率\n\n"
    
    # 半全场TOP3
    htft3_results = [r for r in results if r["htft_top3_hit"] is not None]
    if htft3_results:
        h3 = sum(1 for r in htft3_results if r["htft_top3_hit"])
        report += f"### 6.1 半全场TOP3命中率\n\n"
        report += f"**半全场TOP3命中率**：{h3}/{len(htft3_results)} = **{h3/len(htft3_results):.1%}**\n\n"
    
    # 平系双选
    hf_results = [r for r in results if r["hf_hit"] is not None]
    if hf_results:
        hfh = sum(1 for r in hf_results if r["hf_hit"])
        report += f"### 6.2 平系双选策略命中率（按全场方向对齐）\n\n"
        report += f"**平系双选总体命中率**：{hfh}/{len(hf_results)} = **{hfh/len(hf_results):.1%}**\n"
        report += f"（双选组合：平平+平X，按全场方向对齐）\n\n"
        
        report += "| 平系等级 | 场次 | 命中 | 命中率 | 平均半场平概率 |\n|----------|------|------|--------|---------------|\n"
        for pt in ["严格平系", "均衡平系"]:
            pr = [r for r in hf_results if r["ping_type"] == pt]
            if not pr:
                continue
            ph = sum(1 for r in pr if r["hf_hit"])
            ah = np.mean([r["ht_draw"] for r in pr])
            report += f"| {pt} | {len(pr)} | {ph} | {ph/len(pr):.1%} | {ah:.1%} |\n"
        report += "\n"
        
        report += "### 6.3 按半场平局概率区间统计\n\n"
        report += "| 半场平概率 | 场次 | 双选命中率 |\n|-----------|------|-----------|\n"
        ht_bins = [(0.48,1.0,"≥48%"), (0.45,0.48,"45%~48%"), (0.42,0.45,"42%~45%"), (0,0.42,"<42%")]
        for low, high, label in ht_bins:
            br = [r for r in hf_results if low <= r["ht_draw"] < high]
            if not br:
                continue
            bh = sum(1 for r in br if r["hf_hit"])
            report += f"| {label} | {len(br)} | {bh/len(br):.1%} |\n"
        report += "\n"
    
    # ===== 半全场各双选组合对比（新增）=====
    combo_results = [r for r in results if r["act_htft"] is not None and r["hf_combo_hits"]]
    if combo_results:
        report += "### 6.4 各双选组合命中率对比（全部场次）\n\n"
        report += "| 双选组合 | 场次 | 命中 | 命中率 |\n|---------|------|------|--------|\n"
        combos = ["平平+平胜", "平平+平负", "平平+胜平", "平平+负平",
                  "胜胜+平胜", "负负+平负", "胜胜+平平", "负负+平平"]
        for combo in combos:
            ch = sum(1 for r in combo_results if r["hf_combo_hits"].get(combo, False))
            report += f"| {combo} | {len(combo_results)} | {ch} | {ch/len(combo_results):.1%} |\n"
        report += "\n"
        
        # 按全场方向分组的最优组合
        report += "### 6.5 按全场方向分组的最优双选组合\n\n"
        for dir_val, dir_name in [(0, "全场主胜"), (1, "全场平局"), (2, "全场客胜")]:
            dr = [r for r in combo_results if r["dir_idx"] == dir_val]
            if not dr:
                continue
            report += f"**{dir_name}（{len(dr)}场）**\n\n"
            report += "| 双选组合 | 命中率 |\n|---------|--------|\n"
            for combo in combos:
                ch = sum(1 for r in dr if r["hf_combo_hits"].get(combo, False))
                report += f"| {combo} | {ch/len(dr):.1%} |\n"
            report += "\n"
        
        no_ht = total - len([r for r in results if r["act_htft"] is not None])
        if no_ht > 0:
            report += f"> 📌 有 {no_ht} 场比赛缺少半场比分数据，未纳入半全场回测。\n\n"
    report += "---\n\n"
    
    # ===== 七、回测结论 =====
    report += "## 七、回测结论与优化建议\n\n"
    
    overall_rate = hits / total
    report += f"### 7.1 核心指标汇总\n\n"
    report += "| 指标 | 数值 |\n|------|------|\n"
    report += f"| 胜平负总体命中率 | {overall_rate:.1%} |\n"
    a_results = [r for r in results if r["grade"] == "A档"]
    if a_results:
        report += f"| A档命中率 | {sum(1 for r in a_results if r['hit'])/len(a_results):.1%}（{len(a_results)}场） |\n"
    if full_results:
        report += f"| 含市场赔率融合命中率 | {full_hits/len(full_results):.1%}（{len(full_results)}场有赔率） |\n"
    report += f"| 比分TOP3命中率 | {s3/total:.1%} |\n"
    report += f"| 总进球TOP3命中率 | {t3/total:.1%} |\n"
    report += f"| 大小球准确率 | {o25/total:.1%} |\n"
    if hc_settled:
        report += f"| 让球盘赢盘率（排除走盘） | {hch/len(hc_settled):.1%}（{len(hc_settled)}场已结算） |\n"
    if hf_results:
        report += f"| 平系双选命中率 | {hfh/len(hf_results):.1%}（{len(hf_results)}场） |\n"
    report += "\n"
    
    report += "### 7.2 校准诊断\n\n"
    high_prob = [r for r in results if r["max_prob"] >= 0.5]
    if high_prob:
        hp_actual = sum(1 for r in high_prob if r["hit"]) / len(high_prob)
        hp_pred = np.mean([r["max_prob"] for r in high_prob])
        bias = hp_actual - hp_pred
        report += f"- 高概率区间（≥50%）偏差 **{bias:+.1%}**："
        if bias < -0.05:
            report += "模型**明显高估**强队胜率，建议降低融合概率或增加平局/客胜权重。\n"
        elif bias > 0.05:
            report += "模型**低估**强队胜率，可适当提高强队概率。\n"
        else:
            report += "校准良好，模型概率与实际命中率基本一致。\n"
    
    if hc_settled:
        steady = [r for r in hc_settled if r["handicap"]["win"] >= 0.45]
        if steady:
            sr = sum(1 for r in steady if r["hc_hit"]) / len(steady)
            report += f"- 让球稳胆（预测赢盘≥45%）实际赢盘率 **{sr:.1%}**："
            if sr < 0.40:
                report += "稳胆线偏高，实际赢盘率不足，建议降低稳胆阈值或提高选盘标准。\n"
            else:
                report += "稳胆线合理，实际赢盘率符合预期。\n"
    
    report += "\n### 7.3 优化建议\n\n"
    report += "1. **市场赔率融合**：根据2.2节对比，判断加入市场赔率是否提升准确率，决定是否在每日报告中加大赔率权重。\n"
    report += "2. **分联赛调参**：针对命中率明显偏低的联赛，调整 draw_corr 和融合权重。\n"
    report += "3. **半全场最优组合**：根据6.4/6.5节各组合实际命中率，选择最优双选策略替代当前固定组合。\n"
    report += "4. **让球稳胆阈值**：根据5.1节各档实际赢盘率，验证45%稳胆线是否合理。\n"
    report += "5. **平系阈值优化**：根据6.3节各区间实际命中率，决定严格/均衡阈值是否需要调整。\n"
    report += "6. **大小球校准**：根据4.2节大小球校准偏差，调整总进球预测的联赛修正系数。\n"
    report += "7. **定期回测**：建议每周运行一次，跟踪模型准确率变化趋势。\n\n"
    
    report += "---\n\n"
    report += f"*回测报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # 保存
    os.makedirs("docs", exist_ok=True)
    with open("docs/backtest.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    bt_df = pd.DataFrame([{
        "date": r["date"], "league": r["league"],
        "home": r["home"], "away": r["away"],
        "grade": r["grade"], "pred_prob": r["max_prob"],
        "pred_dir": r["dir_idx"], "actual": r["actual"], "hit": r["hit"],
        "ping_type": r["ping_type"], "ht_draw": r["ht_draw"],
        "hf_hit": r["hf_hit"], "hc_win": r["handicap"]["win"],
        "hc_hit": r["hc_hit"], "hc_draw": r["hc_draw"],
        "is_draw_hc": r["handicap"]["is_draw"],
        "actual_score": r["actual_score"], "actual_tg": r["actual_tg"],
        "score_top3_hit": r["score_top3_hit"], "tg_top3_hit": r["tg_top3_hit"],
        "over25_hit": r["over25_hit"],
    } for r in results])
    bt_df.to_csv("data/backtest_results.csv", index=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 回测摘要")
    print("="*60)
    print(f"  有效样本：{total} 场")
    print(f"  胜平负命中率：{hits/total:.1%}")
    if a_results:
        print(f"  A档命中率：{sum(1 for r in a_results if r['hit'])/len(a_results):.1%}（{len(a_results)}场）")
    if full_results:
        print(f"  含市场赔率融合：{full_hits/len(full_results):.1%}（{len(full_results)}场有赔率）")
    print(f"  比分TOP3命中率：{s3/total:.1%}")
    print(f"  总进球TOP3命中率：{t3/total:.1%}")
    print(f"  大小球准确率：{o25/total:.1%}")
    if hc_settled:
        print(f"  让球赢盘率（排除走盘）：{hch/len(hc_settled):.1%}（走盘{hc_draw_count}场）")
    if hf_results:
        print(f"  平系双选命中率：{hfh/len(hf_results):.1%}（{len(hf_results)}场）")
    print("="*60)
    print(f"\n📄 完整回测报告：docs/backtest.md")
    print(f"📊 回测原始数据：data/backtest_results.csv")

if __name__ == "__main__":
    run_backtest()

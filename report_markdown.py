from typing import List, Dict
from datetime import date

def generate_markdown_report(matches: List[Dict], pingxi_selected: List[Dict], report_date: date) -> str:
    lines = []
    lines.append(f"# 足球预测报告 {report_date}")
    lines.append("")
    lines.append("## 一、今日比赛概览")
    lines.append("")
    
    for i, m in enumerate(matches, 1):
        lines.append(f"### {i}. {m['home_team']} vs {m['away_team']}")
        lines.append(f"- 联赛：{m['league']}")
        lines.append(f"- 主胜：{m['home_win']:.1%} | 平局：{m['draw']:.1%} | 客胜：{m['away_win']:.1%}")
        lines.append(f"- 半场平概率：{m['half_draw']:.1%}（校准后：{m['calibrated_half_draw']:.1%}）")
        lines.append(f"- 平系推荐：{m['pingxi_primary']} + {m['pingxi_secondary']}")
        lines.append("")
    
    lines.append("## 二、平系精选场次")
    lines.append("")
    if pingxi_selected:
        for m in pingxi_selected:
            lines.append(f"- **{m['home_team']} vs {m['away_team']}**：{m['pingxi_primary']} + {m['pingxi_secondary']}，组合概率 {m['combined_prob']:.1%}")
    else:
        lines.append("- 今日无符合标准的平系场次")
    lines.append("")
    
    return "\n".join(lines)

from typing import List, Dict
from datetime import date

def generate_html_report(matches: List[Dict], pingxi_selected: List[Dict], report_date: date) -> str:
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>足球预测报告 {report_date}</title>
        <style>
            body {{ font-family: "微软雅黑", Arial, sans-serif; margin: 20px; background: #f5f7fa; }}
            h1 {{ color: #1a1a2e; text-align: center; }}
            h2 {{ color: #16213e; margin-top: 30px; }}
            .match-card {{ background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
            .match-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #0f3460; }}
            .prob-bar {{ display: flex; height: 24px; border-radius: 12px; overflow: hidden; margin: 8px 0; }}
            .prob-home {{ background: #e94560; color: #fff; text-align: center; line-height: 24px; font-size: 12px; }}
            .prob-draw {{ background: #533483; color: #fff; text-align: center; line-height: 24px; font-size: 12px; }}
            .prob-away {{ background: #0f3460; color: #fff; text-align: center; line-height: 24px; font-size: 12px; }}
            .pingxi-section {{ background: #fff3cd; border-radius: 10px; padding: 18px; margin-top: 20px; }}
            .pingxi-item {{ padding: 8px 0; border-bottom: 1px solid #ffe58f; }}
            .pingxi-item:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <h1>足球预测报告 {report_date}</h1>
        
        <h2>今日比赛预测</h2>
        {''.join([f'''
        <div class="match-card">
            <div class="match-title">{m['home_team']} vs {m['away_team']}</div>
            <div>联赛：{m['league']}</div>
            <div class="prob-bar">
                <div class="prob-home" style="width:{m['home_win']*100:.1f}%">主胜 {m['home_win']:.1%}</div>
                <div class="prob-draw" style="width:{m['draw']*100:.1f}%">平局 {m['draw']:.1%}</div>
                <div class="prob-away" style="width:{m['away_win']*100:.1f}%">客胜 {m['away_win']:.1%}</div>
            </div>
            <div>半场平概率：{m['half_draw']:.1%}（校准后：{m['calibrated_half_draw']:.1%}）</div>
            <div>平系推荐：{m['pingxi_primary']} + {m['pingxi_secondary']}</div>
        </div>
        ''' for m in matches])}
        
        <h2>平系精选</h2>
        <div class="pingxi-section">
            {''.join([f'''
            <div class="pingxi-item">
                <b>{s['home_team']} vs {s['away_team']}</b>：{s['pingxi_primary']} + {s['pingxi_secondary']}，组合概率 {s['combined_prob']:.1%}
            </div>
            ''' for s in pingxi_selected]) if pingxi_selected else '<div>今日无符合标准的平系场次</div>'}
        </div>
    </body>
    </html>
    """
    return html

# 兼容入口：原命令 python predict_report.py 仍可使用
from main import generate_report
from datetime import date
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--format", default="html")
    args = parser.parse_args()
    
    target_date = __import__("datetime").datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    generate_report(target_date, args.format)

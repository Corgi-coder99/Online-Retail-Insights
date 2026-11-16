"""Chay toan bo pipeline theo thu tu: download -> ingest -> clean -> metrics -> forecast -> customer_model -> insights -> report.

Chay: python -I pipeline/run_all.py   (lan dau mat vai phut vi doc file Excel 1 trieu dong)
"""
import subprocess
import sys
from pathlib import Path

STEPS = ["download", "ingest", "clean", "metrics", "forecast", "customer_model", "insights", "report"]
here = Path(__file__).resolve().parent
for s in STEPS:
    print(f"== {s}")
    r = subprocess.run([sys.executable, "-I", str(here / f"{s}.py")])
    if r.returncode:
        sys.exit(f"Buoc {s} loi (ma {r.returncode})")
print("Xong. Mo report/index.html")

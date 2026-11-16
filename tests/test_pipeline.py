"""Kiem tra nhanh sau khi chay pipeline. Chay: python -I tests/test_pipeline.py (khong can pytest). In PASS/FAIL tung muc."""
import csv
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
q = lambda s: con.execute(s).fetchone()[0]
check("raw rows = 1,067,371", q("SELECT count(*) FROM raw_online_retail") == 1067371)
check("two sheets overlap on exactly 22,523 rows (all dropped)", q("SELECT count(*) FROM raw_online_retail") - q("SELECT count(*) FROM clean_lines") == 22523)
check("every cleaned row has one line_type", q("SELECT count(*) FROM clean_lines WHERE line_type IS NULL") == 0)
check("sale lines have positive quantity and price", q("SELECT count(*) FROM clean_lines WHERE line_type='sale' AND (quantity<=0 OR price<=0)") == 0)
check("cancellations come only from C-invoices", q("SELECT count(*) FROM clean_lines WHERE line_type='cancellation' AND invoice NOT LIKE 'C%'") == 0)
check("sale lines = 1,015,071", q("SELECT count(*) FROM clean_lines WHERE line_type='sale'") == 1015071)
check("line_amount = quantity * price", q("SELECT count(*) FROM clean_lines WHERE abs(line_amount - quantity*price) > 1e-9") == 0)

fc = json.loads((ROOT / "outputs" / "forecast.json").read_text(encoding="utf-8"))
check("forecast has 13 weeks", len(fc["forecast"]) == 13)
check("forecast intervals ordered lo <= point <= hi", all(f["lo"] <= f["point"] + 1e-6 <= f["hi"] + 1e-6 for f in fc["forecast"]))
check("chosen model backtest error < 30%", fc["chosen_wape"] < 0.30, f"{fc['chosen_wape']:.3f}")
check("chosen model not worse than seasonal naive", fc["chosen_wape"] <= fc["seasonal_naive_wape"] + 1e-9)

ins = json.loads((ROOT / "outputs" / "insights.json").read_text(encoding="utf-8"))["insights"]
check("13 insights", len(ins) == 13, str(len(ins)))
check("every insight has evidence, confidence, caveat and action", all(all(x.get(k) for k in ("evidence", "confidence", "caveat", "action")) for x in ins))
check("confidence values valid", all(x["confidence"] in ("high", "medium", "low") for x in ins))

bad = []
for p in (ROOT / "outputs" / "tables").glob("*.csv"):
    head = next(csv.reader(open(p, encoding="utf-8")))
    if "customer_id" in head:
        bad.append(p.name)
check("no customer-level table published", not bad, ",".join(bad))
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)

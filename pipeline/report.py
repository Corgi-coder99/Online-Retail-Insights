"""Buoc 6: gop ket qua (bang, du bao, insight) thanh report/index.html mot file doc lap.

Chay: python -I pipeline/report.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    fc = json.loads((OUT / "forecast.json").read_text(encoding="utf-8"))
    ins = json.loads((OUT / "insights.json").read_text(encoding="utf-8"))
    cm = json.loads((OUT / "customer_model.json").read_text(encoding="utf-8"))
    t = OUT / "tables"
    mon = pd.read_csv(t / "monthly.csv", parse_dates=["month"])
    coh = pd.read_csv(t / "cohorts.csv", parse_dates=["cohort", "mon"])
    seg = pd.read_csv(t / "rfm_segments.csv")

    weeks = [pd.Timestamp(h["week"]) for h in fc["history"]]
    ly = {pd.Timestamp(h["week"]): h["y"] for h in fc["history"]}
    forecast = {
        "history": fc["history"], "forecast": fc["forecast"], "forecast_total": fc["forecast_total"], "chosen_model": fc["chosen_model"],
        "chosen_wape": fc["chosen_wape"], "backtest": fc["backtest"],
        "last_year": [ly.get(pd.Timestamp(f["week"]) - pd.Timedelta(weeks=52), 0.0) for f in fc["forecast"]],
    }
    monthly = [{"month": r.month.strftime("%Y-%m-01"), "net": float(r.net_revenue), "orders": int(r.orders), "partial": r.month.strftime("%Y-%m") == "2011-12"} for r in mon.itertuples()]
    cohorts = []
    for c, g in coh.groupby("cohort"):
        g = g.sort_values("k")
        cohorts.append({"cohort": c.strftime("%Y-%m-01"), "size": int(g["size"].iloc[0]), "active": [int(a) for a in g["active"]], "ret": [float(r) for r in g["retention"]]})
    rfm = [{"segment": r.segment, "customers": int(r.customers), "net": float(r.net), "share": float(r.share_of_net), "avg_orders": float(r.avg_orders), "avg_recency_days": float(r.avg_recency_days)} for r in seg.itertuples()]

    byid = {x["id"]: x for x in ins["insights"]}
    g = byid["growth"]["evidence"]
    hero = [
        {"l": "Net revenue, 2 years", "v": f"£{mon['net_revenue'].sum() / 1e6:.1f}M", "d": "Dec 2009 to 9 Dec 2011 (GBP, VAT status unknown)"},
        {"l": "Year-on-year growth", "v": f"{g['growth'] * 100:+.1f}%", "d": f"95% interval {g['ci95'][0] * 100:+.1f}% to {g['ci95'][1] * 100:+.1f}%"},
        {"l": "Next 13 weeks (forecast)", "v": f"£{fc['forecast_total']['point'] / 1e6:.2f}M", "d": f"80% range £{fc['forecast_total']['lo'] / 1e6:.2f}M to £{fc['forecast_total']['hi'] / 1e6:.2f}M"},
        {"l": "Top 10% of customers", "v": f"{byid['concentration']['evidence']['top10_share'] * 100:.0f}%", "d": "of net revenue"},
        {"l": "Revenue with no customer id", "v": f"{byid['tracking_gap']['evidence']['sales_without_customer_id'] / byid['tracking_gap']['evidence']['total_sales'] * 100:.0f}%", "d": "customer metrics cover the rest"},
    ]
    wk = pd.read_csv(t / "weekly_net_revenue.csv")
    spark = {"monthly": [round(float(v), 2) for v in mon["net_revenue"]], "weekly": [round(float(v), 2) for v in wk["net_revenue"]]}
    customer_model = {
        "auc_model": cm["auc_model"], "auc_baseline": cm["auc_baseline"],
        "total_actual": cm["total_actual"], "total_pred_model": cm["total_pred_model"], "total_pred_baseline": cm["total_pred_baseline"],
        "overshoot": cm["total_pred_model"] / cm["total_actual"] - 1,
        "spearman_model": cm["spearman_spend_model"], "spearman_baseline": cm["spearman_spend_baseline"],
        "horizon_days": cm["horizon_days"],
    }
    data = {
        "meta": {"subtitle": "Real, anonymised transactions of a UK online gift retailer, 1 Dec 2009 to 9 Dec 2011. Source: UCI Online Retail II (CC BY 4.0). All code in this repository.",
                 "footer": "Data: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository, DOI 10.24432/C5CG6D, licensed CC BY 4.0. This is an independent analysis, not affiliated with the retailer or UCI. Customer ids are anonymised in the source and no customer-level data is published here. Forecasts are estimates with measured past error, not guarantees."},
        "hero": hero, "forecast": forecast, "monthly": monthly, "cohorts": cohorts, "rfm": rfm,
        "insights": ins["insights"], "spark": spark, "customer_model": customer_model,
    }
    tpl = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float).replace("</", "<\\/")
    out = ROOT / "report" / "index.html"
    theme = (ROOT / "report" / "theme.css").read_text(encoding="utf-8")
    out.write_text(tpl.replace("/*__THEME__*/", theme).replace("/*__DATA__*/", payload), encoding="utf-8")
    print(f"{out} {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

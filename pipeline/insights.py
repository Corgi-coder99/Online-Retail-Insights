"""Buoc 5: "con so biet noi". Bien cac bang so lieu thanh cac phat bieu co bang chung, muc tin cay, gioi han va hanh dong de xuat.

Moi insight co: headline (cau doc duoc), evidence (cac so lieu), confidence (high/medium/low va ly do), caveat, action.
Khong khang dinh nhan qua. Phat bieu nao dua tren du doan deu ghi ro do chinh xac da kiem chung.
Chay: python -I pipeline/insights.py  ->  outputs/insights.json, outputs/insights.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast as fc  # noqa: E402
import metrics as mt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
rng = np.random.default_rng(7)


def gbp(x, d=0):
    a = abs(x)
    s = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{s}GBP {a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{s}GBP {a / 1e3:.0f}K"
    return f"{s}GBP {a:,.{d}f}"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def main():
    con = mt.connect()
    ins = []
    add = lambda **k: ins.append(k)

    series, oneoff = fc.load_series()
    y = series["y"].values
    y1, y2 = y[:52], y[52:104]
    growth = y2.sum() / y1.sum() - 1
    boots = []
    idx = np.arange(52)
    for _ in range(4000):
        b = rng.choice(idx, 52)
        boots.append(y2[b].sum() / y1[b].sum() - 1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    add(id="growth", area="Revenue", headline=f"Net revenue grew {pct(growth)} year on year (weeks 7 Dec 2009 - 5 Dec 2010 vs 6 Dec 2010 - 4 Dec 2011).",
        evidence={"year1_net": float(y1.sum()), "year2_net": float(y2.sum()), "growth": float(growth), "ci95": [float(lo), float(hi)]},
        confidence="medium", why=f"95% bootstrap interval {pct(lo)} to {pct(hi)} (resampling weekly pairs); only two years of data, no external drivers modelled.",
        caveat=f"The weekly series excludes {oneoff[0]} lines above GBP 50,000 (including one 80,995-unit order of GBP 168K and its cancellation). Prices and VAT treatment are not documented in the source.",
        action="Use growth as context, not as a promise: the next-quarter forecast below deliberately does not extrapolate it because the backtest showed no gain from doing so.")

    m = mt.monthly(con)
    m["month"] = pd.to_datetime(m["month"])
    last12 = m[(m["month"] >= "2010-12-01") & (m["month"] <= "2011-11-01")]
    q4 = last12[last12["month"].dt.month.isin([9, 10, 11])]["net_revenue"].sum() / last12["net_revenue"].sum()
    peak = last12.sort_values("net_revenue", ascending=False).iloc[0]
    add(id="seasonality", area="Revenue", headline=f"September to November delivered {pct(q4)} of the last 12 months' net revenue (25% would be flat).",
        evidence={"sep_nov_share": float(q4), "peak_month": peak["month"].strftime("%Y-%m"), "peak_month_net": float(peak["net_revenue"])},
        confidence="high", why="Same pattern in both years of data (see monthly table); effect is large relative to noise.",
        caveat="The business closes for about two weeks over Christmas, so December is short.",
        action="Plan stock, staffing and marketing around a Sep-Nov peak; judge a quarter against the same quarter last year, not the previous one.")

    rfm = mt.customers_rfm(con)
    rfm_sorted = rfm.sort_values("net", ascending=False)
    top10 = rfm_sorted.head(int(len(rfm) * 0.1))["net"].sum() / rfm["net"].sum()
    champ = rfm[rfm["segment"] == "Champions"]
    add(id="concentration", area="Customers", headline=f"The top 10% of customers account for {pct(top10)} of net revenue; the {len(champ):,} 'Champions' alone for {pct(champ['net'].sum() / rfm['net'].sum())}.",
        evidence={"top10_share": float(top10), "champions": int(len(champ)), "champions_share": float(champ["net"].sum() / rfm["net"].sum()), "customers_with_id": int(len(rfm))},
        confidence="high", why="Computed on every customer with an id; no sampling.",
        caveat="Many of these accounts look like trade/wholesale buyers (large baskets), so this is B2B-style concentration, not consumer behaviour.",
        action="Protect these accounts first: named account management, stock priority, early notice of new lines. Losing a handful would move revenue visibly.")

    tot_sales = float(con.execute("SELECT sum(line_amount) FROM clean_lines WHERE line_type='sale'").fetchone()[0])
    no_id = float(con.execute("SELECT sum(line_amount) FROM clean_lines WHERE line_type='sale' AND customer_id IS NULL").fetchone()[0])
    add(id="tracking_gap", area="Data quality", headline=f"{pct(no_id / tot_sales)} of sales revenue (GBP {no_id / 1e6:.2f}M) has no customer id, so customer metrics cover only {pct(1 - no_id / tot_sales)} of revenue.",
        evidence={"sales_without_customer_id": no_id, "total_sales": tot_sales},
        confidence="high", why="Direct count from the source.",
        caveat="Missing ids are probably guest or marketplace orders; they are included in all revenue and product figures.",
        action="Capture a customer or email key at checkout for every order; until then treat RFM, cohort and LTV numbers as a lower bound on customer value.")

    ret = float(-con.execute("SELECT sum(line_amount) FROM clean_lines WHERE line_type='cancellation' AND abs(line_amount) <= 50000").fetchone()[0])
    gross = float(con.execute("SELECT sum(line_amount) FROM clean_lines WHERE line_type='sale' AND abs(line_amount) <= 50000").fetchone()[0])
    prod = mt.products(con)
    prod["ret_rate"] = prod["returns"] / prod["gross_sales"]
    prod["description"] = prod["description"].fillna("").str.split().str.join(" ")
    # chi xet san pham ban nhieu (>= 20K GBP, >= 100 don) va ti le tra < 100% (ti le >= 100% nghia la huy don cua ban ra truoc thoi diem du lieu)
    worst = prod[(prod["gross_sales"] > 20000) & (prod["orders"] >= 100) & (prod["ret_rate"] < 1)].sort_values("ret_rate", ascending=False).head(3)
    add(id="returns", area="Operations", headline=f"Cancellations and returns equal {pct(ret / gross)} of gross sales (GBP {ret / 1e6:.2f}M); among products with over GBP 20K sales and 100 orders, the highest return rates are {', '.join(f'{d.title()} ({pct(r, 0)})' for d, r in zip(worst['description'], worst['ret_rate']))}.",
        evidence={"returns": ret, "gross_sales": gross, "rate": ret / gross, "worst_products": [{"description": d, "rate": float(r)} for d, r in zip(worst["description"], worst["ret_rate"])]},
        confidence="medium", why="Cancellation invoices are not linked to the original sale, so product return rates are approximate.",
        caveat="Excludes the one-off 80,995-unit order of GBP 168K (placed Jan 2011, cancelled Dec 2011).", action="Review the highest-return products for quality, description or packaging problems before reordering.")

    p = prod.sort_values("gross_sales", ascending=False).reset_index(drop=True)
    cum = p["gross_sales"].cumsum() / p["gross_sales"].sum()
    n20 = int(np.ceil(len(p) * 0.2))
    xmas = p[p["description"].fillna("").str.contains("CHRISTMAS|XMAS|ADVENT", regex=True)]
    add(id="products", area="Products", headline=f"The top 20% of products ({n20:,} of {len(p):,}) generate {pct(cum.iloc[n20 - 1])} of gross sales; the best seller is '{p['description'].iloc[0].title()}' ({gbp(p['gross_sales'].iloc[0])}).",
        evidence={"products_sold": int(len(p)), "top20_share": float(cum.iloc[n20 - 1]), "top_product": p["description"].iloc[0], "christmas_named_share": float(xmas["gross_sales"].sum() / p["gross_sales"].sum())},
        confidence="high", why="Complete product table.", caveat="A long tail is normal for gift ware; the value is in range management, not in cutting the tail blindly.",
        action="Review the bottom of the tail each season for products with few buyers and low sales, and keep depth on the proven top lines.")

    co = mt.countries(con)
    uk = co[co["country"] == "United Kingdom"].iloc[0]
    exp = co[(co["country"] != "United Kingdom") & (co["y1"] > 30000)].copy()
    exp["growth"] = exp["y2"] / exp["y1"] - 1
    top_exp = exp.sort_values("y2", ascending=False).head(5)
    fast = exp.sort_values("growth", ascending=False).head(3)
    add(id="geography", area="Markets", headline=f"The UK is {pct(uk['gross_sales'] / co['gross_sales'].sum())} of gross sales; the fastest-growing export markets with over GBP 30K in year 1 are {', '.join(f'{c} (+{g * 100:.0f}%)' for c, g in zip(fast['country'], fast['growth']))}.",
        evidence={"uk_share": float(uk["gross_sales"] / co["gross_sales"].sum()), "fast": fast[["country", "y1", "y2", "growth"]].round(3).to_dict(orient="records")},
        confidence="low", why="Year-1 vs year-2 comparison in single markets rests on a few large accounts.",
        caveat="Growth rates in small markets are volatile; a single customer can explain them.", action="Check whether growth comes from one account before investing in a market.")

    pairs = mt.basket_pairs(con)
    pairs = pairs[(pairs["together"] >= 300)].sort_values("lift", ascending=False).head(3)
    add(id="bundles", area="Products", headline="Strong bundle candidates (items that are bought together far more than chance): " + "; ".join(f"{' '.join(a.split()).title()} + {' '.join(b.split()).title()} (lift {l:.0f}, together in {n:,} orders)" for a, b, l, n in zip(pairs["desc_a"], pairs["desc_b"], pairs["lift"], pairs["together"])) + ".",
        evidence={"pairs": pairs[["desc_a", "desc_b", "together", "lift", "confidence_a_to_b", "confidence_b_to_a"]].round(3).to_dict(orient="records")},
        confidence="high", why="Pair counts of at least 300 orders each; lift above 30 means colour or theme variants of one range.",
        caveat="High lift here mostly shows range variants bought as a set by trade buyers, which is not the same as cross-selling unrelated items.",
        action="Offer these as ready-made sets or multi-buy prices and check basket size before and after.")

    c = mt.cohorts(con)
    k1 = c[(c["k"] == 1) & (c["cohort"] <= "2011-10-01")]["retention"]
    d12 = c[(c["cohort"] <= "2010-11-01") & (c["k"] <= 11)].groupby("cohort")["rev_per_customer"].sum()
    k12 = c[(c["cohort"] <= "2010-11-01") & (c["k"] == 12)]["retention"]
    add(id="retention", area="Customers", headline=f"Month-1 retention averages {pct(k1.mean())} across cohorts; customers who started between Dec 2009 and Nov 2010 spent a median of {gbp(d12.median())} in their first 12 months, and {pct(k12.mean())} were still buying in month 12.",
        evidence={"month1_retention_mean": float(k1.mean()), "ltv12_median": float(d12.median()), "month12_retention_mean": float(k12.mean()), "cohorts": int(len(d12))},
        confidence="medium", why=f"{len(d12)} full-year cohorts of {int(c[(c['k'] == 0) & (c['cohort'] <= '2010-11-01')]['size'].min()):,} to {int(c[(c['k'] == 0) & (c['cohort'] <= '2010-11-01')]['size'].max()):,} customers each; the later the cohort, the fewer the customers.", caveat="Customers without an id are excluded; first-order month includes some existing customers from before the dataset starts (Dec 2009 cohort).",
        action="Treat 12-month spend per new customer as the ceiling for acquisition cost; aim retention efforts at the first 3 months where most customers drop.")

    seg = rfm.groupby("segment").agg(customers=("customer_id", "count"), net=("net", "sum")).reset_index()
    ar = seg[seg["segment"] == "At risk"].iloc[0]
    add(id="winback", area="Customers", headline=f"{int(ar['customers']):,} 'At risk' customers (3+ orders but silent for a long time) bought {gbp(ar['net'])} in total; they are the clearest win-back list.",
        evidence={"at_risk_customers": int(ar["customers"]), "at_risk_net": float(ar["net"]), "segments": seg.round(0).to_dict(orient="records")},
        confidence="medium", why="Segments come from quintile scores; the At-risk rule is recency 1-2 and frequency 3+.", caveat="Quintile rules are a judgement call; no test has yet shown that contacting them lifts sales.",
        action="Run a holdout test: contact half of the list with a personal offer, compare orders over 90 days with the other half.")

    fj = json.loads((OUT / "forecast.json").read_text(encoding="utf-8"))
    ft = fj["forecast_total"]
    cw = next(r for r in fj["backtest"] if r["model"] == fj["chosen_model"])
    add(id="forecast_13w", area="Forecast", headline=f"Expect about {gbp(ft['point'])} of net revenue in the 13 weeks from 5 Dec 2011 (80% range {gbp(ft['lo'])} to {gbp(ft['hi'])}).",
        evidence={"point": ft["point"], "lo": ft["lo"], "hi": ft["hi"], "same_period_last_year": fj["same_period_last_year"], "model": fj["chosen_model"], "backtest_wape": fj["chosen_wape"], "backtest_coverage_80": fj["backtest_coverage_80"], "models": fj["backtest"]},
        confidence="medium", why=f"Chosen model: {fj['chosen_model']} (same weeks last year). Backtest over 9 rolling origins: weighted error {pct(fj['chosen_wape'])}; the 80% range covered {pct(fj['backtest_coverage_80'], 0)} of backtest weeks. Regression and ensemble models were not clearly better (needed 2 points; got 0.1), so the simpler method was kept.",
        caveat="Two years of history: one Christmas season cannot prove a trend. The year-over-year growth is not added because adding it made the backtest worse.", action="Use the range for planning cash and stock; re-run weekly and compare with actuals to see if the range holds.")

    cm = json.loads((OUT / "customer_model.json").read_text(encoding="utf-8"))
    fut = cm["future"]
    scale = cm["total_actual"] / cm["total_pred_model"]
    add(id="customer_forecast", area="Forecast", headline=f"About {fut['expected_buyers']:,.0f} of {fut['customers']:,} known customers ({pct(fut['expected_buyers'] / fut['customers'], 0)}) are expected to buy again within 180 days of 10 Dec 2011, worth roughly {gbp(fut['expected_spend_180d'] * scale)} to {gbp(fut['expected_spend_180d'])}.",
        evidence={"expected_buyers": fut["expected_buyers"], "customers": fut["customers"], "expected_spend_raw": fut["expected_spend_180d"], "expected_spend_calibrated": fut["expected_spend_180d"] * scale,
                  "auc_model": cm["auc_model"], "auc_baseline": cm["auc_baseline"], "top_decile_capture_model": cm["top_decile_capture_model"], "top_decile_capture_baseline": cm["top_decile_capture_baseline"],
                  "backtest_total_ratio": cm["total_pred_model"] / cm["total_actual"], "segments": fut["segments"]},
        confidence="low", why=f"On the 2011-06-09 backtest the model ranked customers about as well as 'recent activity' (AUC {cm['auc_model']:.2f} vs {cm['auc_baseline']:.2f}; top decile captured {pct(cm['top_decile_capture_model'], 0)} vs {pct(cm['top_decile_capture_baseline'], 0)} of spend) and over-predicted the total by {pct(cm['total_pred_model'] / cm['total_actual'] - 1, 0)}; the lower figure applies that correction.",
        caveat="Do not use individual probabilities for decisions; use segment totals, and prefer the time-series forecast for company-level revenue.", action="Use as a sizing range for the retention budget, then replace it with a measured holdout test.")

    hv = fut["high_value_low_prob"]
    add(id="value_at_risk", area="Forecast", headline=f"{hv['customers']} of the {hv['of_high_value']:,} highest-spending customers (top 20% by lifetime sales) have a low modelled chance (<35%) of buying again; together they have bought {gbp(hv['hist_gross'])}.",
        evidence=hv, confidence="low", why="Same model as above, so same limits.", caveat="A small group; verify account by account before acting.",
        action="Have account managers call these customers personally; the cost is small compared with the value.")

    res = {"generated_from": "UCI Online Retail II, 2009-12-01 to 2011-12-09 (GBP)", "n_insights": len(ins), "insights": ins}
    OUT.mkdir(exist_ok=True)
    (OUT / "insights.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    lines = ["# Insights (auto-generated)", "", "Generated by `pipeline/insights.py` from the cleaned data. Each statement lists its evidence, a confidence level with the reason, a caveat, and a suggested action.", ""]
    for i, x in enumerate(ins, 1):
        lines += [f"## {i}. {x['headline']}", "", f"- **Area:** {x['area']}  |  **Confidence:** {x['confidence']}", f"- **Why this confidence:** {x['why']}", f"- **Caveat:** {x['caveat']}", f"- **Suggested action:** {x['action']}", ""]
    (OUT / "insights.md").write_text("\n".join(lines), encoding="utf-8")
    for x in ins:
        print(f"[{x['confidence']:6s}] {x['id']:18s} {x['headline'][:150]}")


if __name__ == "__main__":
    main()

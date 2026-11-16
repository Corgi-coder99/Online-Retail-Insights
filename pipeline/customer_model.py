"""Buoc 4b: du bao hanh vi khach hang 180 ngay toi (co mua lai khong, se chi bao nhieu) va chat luong cua du bao do.

Cach kiem chung theo thoi gian (khong ro ri): huan luyen tren cac moc cat 2010 (muc tieu da biet), kiem tra tren moc cat 2011-06-09
(muc tieu 2011-06-10..2011-12-09 da biet). Moc cat 2011-12-09 la du bao that cho tuong lai, khong co dap an.
So sanh voi mo hinh co so: "180 ngay toi se chi bang 180 ngay vua qua" va xep hang theo so don/recency.
Khong xuat du lieu tung khach hang, chi so lieu tong hop.

Chay: python -I pipeline/customer_model.py  ->  outputs/customer_model.json
"""
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")  # mot luong de ket qua lap lai duoc (cong don so thuc da luong co the lech nhe)
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "warehouse.duckdb"
OUT = ROOT / "outputs"
HORIZON = 180
FEATS = ["recency_days", "orders", "tenure_days", "gross_to_date", "aov", "orders_90d", "orders_180d", "spend_180d", "return_ratio",
         "products", "avg_gap_days", "is_uk"]


def snapshot(con, cutoff, with_target):
    cut = pd.Timestamp(cutoff)
    end = cut + pd.Timedelta(days=HORIZON)
    f = con.execute("""
        WITH s AS (SELECT customer_id, invoice, min(invoice_ts) ts, sum(line_amount) amt, any_value(country) country
                   FROM clean_lines WHERE line_type='sale' AND customer_id IS NOT NULL AND invoice_ts < ? GROUP BY 1, 2),
             r AS (SELECT customer_id, -sum(line_amount) ret FROM clean_lines WHERE line_type='cancellation' AND customer_id IS NOT NULL AND invoice_ts < ? GROUP BY 1),
             p AS (SELECT customer_id, count(DISTINCT stock_code) products FROM clean_lines WHERE line_type='sale' AND customer_id IS NOT NULL AND invoice_ts < ? GROUP BY 1)
        SELECT s.customer_id, min(ts) first_ts, max(ts) last_ts, count(*) orders, sum(amt) gross_to_date,
               count(*) FILTER (WHERE ts >= ? - INTERVAL 90 DAY) orders_90d,
               count(*) FILTER (WHERE ts >= ? - INTERVAL 180 DAY) orders_180d,
               coalesce(sum(amt) FILTER (WHERE ts >= ? - INTERVAL 180 DAY), 0) spend_180d,
               coalesce(any_value(r.ret), 0) ret, any_value(p.products) products,
               max(country = 'United Kingdom') is_uk
        FROM s LEFT JOIN r USING (customer_id) LEFT JOIN p USING (customer_id) GROUP BY s.customer_id""",
                    [cut] * 3 + [cut] * 3).df()
    f = f.sort_values("customer_id").reset_index(drop=True)  # thu tu co dinh de ket qua lap lai duoc
    f["recency_days"] = (cut - f["last_ts"]).dt.days
    f["tenure_days"] = (f["last_ts"] - f["first_ts"]).dt.days
    f["aov"] = f["gross_to_date"] / f["orders"]
    f["return_ratio"] = f["ret"] / f["gross_to_date"].clip(lower=1)
    f["avg_gap_days"] = np.where(f["orders"] > 1, f["tenure_days"] / (f["orders"] - 1), np.nan)
    f["is_uk"] = f["is_uk"].astype(int)
    f["cutoff"] = cut
    for col in ("gross_to_date", "aov", "spend_180d", "return_ratio", "avg_gap_days", "ret"):
        f[col] = f[col].astype(float).round(4)  # lam tron de sai so cong don so thuc cua DuckDB khong doi nhanh cay
    if with_target:
        t = con.execute("""SELECT customer_id, sum(line_amount) spend, count(DISTINCT invoice) n FROM clean_lines
                           WHERE line_type='sale' AND customer_id IS NOT NULL AND invoice_ts >= ? AND invoice_ts < ? GROUP BY 1""", [cut, end]).df()
        f = f.merge(t, on="customer_id", how="left")
        f["spend"] = f["spend"].fillna(0.0)
        f["bought"] = (f["n"].fillna(0) > 0).astype(int)
    return f


def decile_table(pred, actual, k=10):
    d = pd.DataFrame({"pred": pred, "actual": actual})
    d["bin"] = pd.qcut(d["pred"].rank(method="first"), k, labels=False)
    g = d.groupby("bin").agg(n=("pred", "size"), predicted=("pred", "sum"), actual=("actual", "sum")).reset_index()
    return g


def main():
    con = duckdb.connect(str(DB), read_only=True)
    train_cuts, test_cut, final_cut = ["2010-04-09", "2010-06-09"], "2011-06-09", "2011-12-10"
    tr = pd.concat([snapshot(con, c, True) for c in train_cuts], ignore_index=True)
    te = snapshot(con, test_cut, True)
    print(f"train: {len(tr):,} dong-khach (moc {train_cuts}); test: {len(te):,} khach (moc {test_cut})")

    reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=0, early_stopping=False).fit(tr[FEATS], np.log1p(tr["spend"]))
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=0, early_stopping=False).fit(tr[FEATS], tr["bought"])
    te["pred_spend"] = np.expm1(reg.predict(te[FEATS])).clip(min=0)
    te["p_buy"] = clf.predict_proba(te[FEATS])[:, 1]
    # CLV mong doi = xac suat mua x chi tieu neu mua; huan luyen rieng chi tieu khi co mua de tranh trung binh bi keo ve 0
    reg_pos = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=0, early_stopping=False).fit(tr[tr["bought"] == 1][FEATS], np.log1p(tr[tr["bought"] == 1]["spend"]))
    te["exp_spend"] = te["p_buy"] * np.expm1(reg_pos.predict(te[FEATS])).clip(min=0)

    base_spend = te["spend_180d"].values  # co so: chi tieu 180 ngay toi = chi tieu 180 ngay vua qua
    base_rank = (te["orders_180d"] * 1000 - te["recency_days"]).values  # co so xep hang: so don 180 ngay roi recency

    res = {
        "cutoffs": {"train": train_cuts, "test": test_cut, "final": final_cut}, "horizon_days": HORIZON,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_actual_repeat_rate": float(te["bought"].mean()),
        "auc_model": float(roc_auc_score(te["bought"], te["p_buy"])),
        "auc_baseline": float(roc_auc_score(te["bought"], base_rank)),
        "spearman_spend_model": float(spearmanr(te["exp_spend"], te["spend"]).statistic),
        "spearman_spend_baseline": float(spearmanr(base_spend, te["spend"]).statistic),
        "total_actual": float(te["spend"].sum()), "total_pred_model": float(te["exp_spend"].sum()), "total_pred_baseline": float(base_spend.sum()),
    }
    # top 10% theo du bao nam giu bao nhieu % chi tieu thuc te
    k = max(1, int(0.1 * len(te)))
    for name, score in (("model", te["exp_spend"].values), ("baseline", base_spend)):
        top = np.argsort(-score)[:k]
        res[f"top_decile_capture_{name}"] = float(te["spend"].values[top].sum() / te["spend"].sum())
    res["calibration_deciles"] = decile_table(te["exp_spend"], te["spend"]).round(0).to_dict(orient="records")
    print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in res.items() if k_ != "calibration_deciles"})

    # du bao that: huan luyen tren moi moc da biet muc tieu, ap len moc 2011-12-10
    allcuts = train_cuts + [test_cut, "2011-04-09", "2010-12-10"]  # them moc cuoi nam 2010 de mo hinh thay cung mua vu voi moc du bao 2011-12-10
    full = pd.concat([snapshot(con, c, True) for c in allcuts], ignore_index=True)
    fut = snapshot(con, final_cut, False)
    clf2 = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=0, early_stopping=False).fit(full[FEATS], full["bought"])
    pos = full[full["bought"] == 1]
    reg2 = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=0, early_stopping=False).fit(pos[FEATS], np.log1p(pos["spend"]))
    fut["p_buy"] = clf2.predict_proba(fut[FEATS])[:, 1]
    fut["exp_spend"] = fut["p_buy"] * np.expm1(reg2.predict(fut[FEATS])).clip(min=0)
    # phan khuc RFM hien tai de gop (dung dinh nghia metrics.py)
    import metrics
    rfm = metrics.customers_rfm(con, pd.Timestamp(final_cut)).set_index("customer_id")["segment"]
    fut["segment"] = fut["customer_id"].map(rfm)
    seg = fut.groupby("segment").agg(customers=("customer_id", "size"), expected_buyers=("p_buy", "sum"), expected_spend=("exp_spend", "sum"),
                                     hist_net_gross=("gross_to_date", "sum")).reset_index()
    seg["avg_p_buy"] = seg["expected_buyers"] / seg["customers"]
    res["future"] = {"snapshot": final_cut, "customers": int(len(fut)), "expected_buyers": float(fut["p_buy"].sum()),
                     "expected_spend_180d": float(fut["exp_spend"].sum()), "segments": seg.round(3).to_dict(orient="records")}
    # gia tri co nguy co mat: khach co p_buy thap nhung chi tieu lich su cao (top 20% theo gross)
    hv = fut[fut["gross_to_date"] >= fut["gross_to_date"].quantile(0.8)]
    low = hv[hv["p_buy"] < 0.35]
    res["future"]["high_value_low_prob"] = {"customers": int(len(low)), "of_high_value": int(len(hv)), "hist_gross": float(low["gross_to_date"].sum())}
    OUT.mkdir(exist_ok=True)
    (OUT / "customer_model.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("du bao 180 ngay (tu 2011-12-10):", {k_: (round(v, 1) if isinstance(v, float) else v) for k_, v in res["future"].items() if k_ not in ("segments",)})
    print(seg.round(2).to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

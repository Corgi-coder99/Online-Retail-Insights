"""Buoc 4a: du bao doanh thu rong theo tuan 13 tuan toi, co kiem chung nguoc (backtest) va khoang tin cay.

Nguyen tac: mo hinh phuc tap chi duoc dung neu thang mo hinh don gian (seasonal naive) trong backtest.
Chuoi: doanh thu rong theo tuan (da bo 2 dong ngoai le: don 80.995 san pham va lenh huy cua no).
Chay: python -I pipeline/forecast.py   ->  outputs/forecast.json, outputs/tables/forecast_backtest.csv
"""
import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "warehouse.duckdb"
OUT = ROOT / "outputs"
H = 13  # tuan du bao


def load_series():
    con = duckdb.connect(str(DB), read_only=True)
    oneoff = con.execute("SELECT count(*), sum(line_amount) FROM clean_lines WHERE line_type IN ('sale','cancellation') AND abs(line_amount) > 50000").fetchone()
    df = con.execute("""
        SELECT date_trunc('week', invoice_date) AS week, sum(line_amount) AS y, count(DISTINCT invoice_date) AS trading_days
        FROM clean_lines WHERE line_type IN ('sale','cancellation') AND abs(line_amount) <= 50000 GROUP BY 1 ORDER BY 1""").df()
    con.close()
    df = df.set_index("week")
    full = pd.date_range(df.index.min(), df.index.max(), freq="7D")
    df = df.reindex(full).rename_axis("week")
    df["y"] = df["y"].fillna(0.0)
    df["trading_days"] = df["trading_days"].fillna(0).astype(int)
    df = df.iloc[1:-1]  # bo tuan dau va tuan cuoi khong tron (du lieu bat dau thu ba 1/12/2009, ket thuc thu sau 9/12/2011)
    return df, oneoff


def features(index, closure_weeks, K=3):
    t = np.arange(len(index)).astype(float)
    doy = index.dayofyear.values + 3  # lay giua tuan
    X = pd.DataFrame({"const": 1.0, "trend": t / 52.0}, index=index)
    for k in range(1, K + 1):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    X["closure"] = index.isin(closure_weeks).astype(float)
    return X


def closure_weeks_from(df):
    """Tuan nghi le cuoi nam: tuan co it hon 3 ngay giao dich, trong khoang 20/12 den 8/1."""
    cw = []
    for w, r in df.iterrows():
        mid = w + pd.Timedelta(days=3)
        if r["trading_days"] < 3 and ((mid.month == 12 and mid.day >= 20) or (mid.month == 1 and mid.day <= 8)):
            cw.append(w)
    return pd.DatetimeIndex(cw)


def fit_predict_reg(train, future_index, closure):
    """Hoi quy log(y) theo xu huong + Fourier (K=3) + tuan nghi le. y = 0 trong tuan nghi duoc xu ly bang bien closure."""
    tr = train[train["y"] > 0]
    X = features(train.index, closure).loc[tr.index]
    model = sm.OLS(np.log(tr["y"]), X).fit()
    Xf = features(train.index.append(future_index), closure).loc[future_index]
    mu = model.predict(Xf)
    resid_sd = float(np.std(model.resid, ddof=X.shape[1]))
    return np.exp(mu + 0.5 * resid_sd ** 2).values, resid_sd


def seasonal_naive(train, future_index):
    out = []
    for w in future_index:
        ly = w - pd.Timedelta(weeks=52)
        out.append(train["y"].get(ly, np.nan))
    return np.array(out, dtype=float)


def forecast_models(train, future_index, closure):
    sn = seasonal_naive(train, future_index)
    reg, sd = fit_predict_reg(train, future_index, closure)
    # nam ngoai quan sat cuoi: ti le tang truong 13 tuan gan nhat so voi cung ky nam truoc, ap len seasonal naive
    # tang truong 26 tuan gan nhat so voi cung ky nam truoc, co chan [0.8, 1.25] va co ve 50% ve 1 (tranh phong dai nhieu)
    n = 26
    last = train["y"].iloc[-n:].sum(); ly = train["y"].reindex(train.index[-n:] - pd.Timedelta(weeks=52)).sum()
    raw_growth = last / ly if ly > 0 else 1.0
    growth = 1.0 + 0.5 * (min(max(raw_growth, 0.8), 1.25) - 1.0)
    sng = sn * growth
    ens = np.nanmean(np.vstack([sn, reg]), axis=0)  # trung binh seasonal naive va hoi quy
    return {"naive_last_value": np.repeat(float(train["y"].iloc[-1]), len(future_index)), "seasonal_naive": sn,
            "seasonal_naive_x_growth": sng, "regression_fourier": reg, "ensemble": ens}, {"growth": float(growth), "raw_growth": float(raw_growth), "resid_sd": sd}


def main():
    df, oneoff = load_series()
    closure = closure_weeks_from(df)
    print(f"series: {len(df)} tuan, {df.index.min().date()} .. {df.index.max().date()}; tuan nghi le: {[d.date().isoformat() for d in closure]}")
    # backtest rolling origin: goc bat dau o tuan thu 56 (can >= 52 tuan de co nam truoc), cach nhau 4 tuan
    rows = []
    for o in range(56, len(df) - H + 1, 4):
        train = df.iloc[:o]; fut = df.index[o:o + H]; actual = df["y"].iloc[o:o + H].values
        preds, _ = forecast_models(train, fut, closure)
        for name, p in preds.items():
            ok = ~np.isnan(p)
            rows.append({"origin": train.index[-1].date().isoformat(), "model": name, "n": int(ok.sum()),
                         "abs_err": float(np.abs(p[ok] - actual[ok]).sum()), "actual": float(actual[ok].sum()),
                         "pred_sum": float(p[ok].sum()), "sq": float(((p[ok] - actual[ok]) ** 2).sum())})
    bt = pd.DataFrame(rows)
    sc = bt.groupby("model").agg(origins=("origin", "nunique"), wape=("abs_err", "sum"), actual=("actual", "sum"), bias=("pred_sum", "sum")).reset_index()
    sc["wape"] = sc["wape"] / sc["actual"]; sc["bias"] = sc["bias"] / sc["actual"] - 1
    sc["wape_pct"] = (sc["wape"] * 100).round(1)
    sc = sc.sort_values("wape")
    print(sc[["model", "origins", "wape_pct", "bias"]].round(3).to_string(index=False))
    # chon mo hinh tot nhat theo WAPE nhung chi chap nhan mo hinh phuc tap neu hon seasonal naive
    best = sc.iloc[0]["model"]
    base = float(sc[sc["model"] == "seasonal_naive"]["wape"].iloc[0])
    # quy tac chon: mo hinh phuc tap phai thang seasonal naive it nhat 2 diem phan tram WAPE, neu khong thi dung mo hinh don gian
    if best != "seasonal_naive" and base - float(sc.iloc[0]["wape"]) < 0.02:
        print(f"{best} chi hon seasonal naive {100 * (base - float(sc.iloc[0]['wape'])):.1f} diem % (< 2): giu mo hinh don gian")
        best = "seasonal_naive"
    # khoang tin cay 80% tu ti le sai so backtest (actual / forecast theo tung tuan) cua mo hinh duoc chon
    ratios = []
    for o in range(56, len(df) - H + 1, 4):
        train = df.iloc[:o]; fut = df.index[o:o + H]
        p = forecast_models(train, fut, closure)[0][best]; a = df["y"].iloc[o:o + H].values
        m = (p > 0) & ~np.isnan(p) & (a > 0)
        ratios += list(a[m] / p[m])
    lo, hi = np.quantile(ratios, [0.10, 0.90])
    # du bao that su cho 13 tuan sau du lieu
    fut_idx = pd.date_range(df.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="7D")
    # tuan nghi le tuong lai: tuan chua ngay 25/12 va tuan chua ngay 1/1
    future_closure = closure.append(pd.DatetimeIndex([w for w in fut_idx if any(
        (w <= pd.Timestamp(d) < w + pd.Timedelta(days=7)) for d in ("2011-12-27", "2012-01-02"))]))
    preds, info = forecast_models(df, fut_idx, future_closure)
    pt = preds[best]
    # backtest coverage cua khoang 80%
    cov = float(np.mean([(r >= lo) and (r <= hi) for r in ratios]))
    result = {
        "series_start": df.index.min().date().isoformat(), "series_end": (df.index.max() + pd.Timedelta(days=6)).date().isoformat(),
        "horizon_weeks": H, "one_off_lines_removed": {"lines": int(oneoff[0]), "amount_gbp": float(oneoff[1])},
        "backtest": sc[["model", "origins", "wape", "bias"]].to_dict(orient="records"),
        "chosen_model": best, "chosen_wape": float(sc[sc["model"] == best]["wape"].iloc[0]), "seasonal_naive_wape": base,
        "interval_80_ratio": [float(lo), float(hi)], "backtest_coverage_80": cov, "growth_vs_last_year_26w": info["raw_growth"], "growth_applied": info["growth"],
        "history": [{"week": d.date().isoformat(), "y": float(v)} for d, v in df["y"].items()],
        "forecast": [{"week": d.date().isoformat(), "point": float(p), "lo": float(p * lo), "hi": float(p * hi)} for d, p in zip(fut_idx, pt)],
        "forecast_total": {"point": float(pt.sum()), "lo": float(pt.sum() * lo), "hi": float(pt.sum() * hi)},
        "same_period_last_year": float(df["y"].reindex(fut_idx - pd.Timedelta(weeks=52)).sum()),
        "all_models_total": {k: float(np.nansum(v)) for k, v in preds.items()},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "forecast.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    bt.to_csv(OUT / "tables" / "forecast_backtest.csv", index=False)
    print(f"mo hinh chon: {best}; WAPE {float(sc[sc['model'] == best]['wape'].iloc[0]):.1%} (seasonal naive {base:.1%})")
    print(f"13 tuan toi: {pt.sum():,.0f} GBP (80%: {pt.sum() * lo:,.0f} .. {pt.sum() * hi:,.0f}); cung ky nam truoc {result['same_period_last_year']:,.0f}")
    print(f"coverage 80% tren backtest: {cov:.0%}")


if __name__ == "__main__":
    main()

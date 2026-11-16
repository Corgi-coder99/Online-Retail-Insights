"""Buoc 3: tinh cac bang so lieu (thang, tuan, khach hang RFM, cohort, san pham, cap san pham mua cung nhau, quoc gia).

Dung lam thu vien cho insights.py, forecast.py va report.py; chay truc tiep de ghi CSV vao outputs/tables.
Doanh thu rong = ban hang (sale) tru hang tra (cancellation) cua san pham that. GBP, chua ro VAT (nguon khong ghi).

Chay: python -I pipeline/metrics.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "warehouse.duckdb"
SNAPSHOT = pd.Timestamp("2011-12-10")  # ngay sau ban ghi cuoi cung (2011-12-09)


def connect():
    return duckdb.connect(str(DB), read_only=True)


def monthly(con):
    return con.execute("""
        SELECT date_trunc('month', invoice_date) AS month,
               sum(CASE WHEN line_type='sale' THEN line_amount END) AS gross_sales,
               -sum(CASE WHEN line_type='cancellation' THEN line_amount END) AS returns,
               sum(CASE WHEN line_type IN ('sale','cancellation') THEN line_amount END) AS net_revenue,
               count(DISTINCT CASE WHEN line_type='sale' THEN invoice END) AS orders,
               count(DISTINCT CASE WHEN line_type='sale' THEN customer_id END) AS customers,
               max(invoice_date) AS last_day
        FROM clean_lines WHERE line_type IN ('sale','cancellation') GROUP BY 1 ORDER BY 1""").df()


def weekly(con):
    """Doanh thu rong theo tuan (Thu hai dau tuan); bo tuan dau va tuan cuoi khong tron."""
    df = con.execute("""
        SELECT date_trunc('week', invoice_date) AS week,
               sum(line_amount) AS net_revenue,
               count(DISTINCT CASE WHEN line_type='sale' THEN invoice END) AS orders,
               count(DISTINCT invoice_date) AS trading_days
        FROM clean_lines WHERE line_type IN ('sale','cancellation') GROUP BY 1 ORDER BY 1""").df()
    return df.iloc[1:-1].reset_index(drop=True)  # tuan dau (bat dau thu ba 1/12/2009) va tuan cuoi (den 9/12/2011) khong tron


def invoices(con):
    return con.execute("""
        SELECT invoice, any_value(customer_id) AS customer_id, min(invoice_ts) AS ts, any_value(country) AS country,
               sum(line_amount) AS gross, sum(quantity) AS units, count(*) AS lines
        FROM clean_lines WHERE line_type='sale' GROUP BY invoice""").df()


def customers_rfm(con, snapshot=SNAPSHOT):
    df = con.execute("""
        WITH s AS (SELECT customer_id, invoice, min(invoice_ts) ts, sum(line_amount) amt FROM clean_lines
                   WHERE line_type='sale' AND customer_id IS NOT NULL AND invoice_ts < ? GROUP BY 1, 2),
             r AS (SELECT customer_id, -sum(line_amount) ret FROM clean_lines
                   WHERE line_type='cancellation' AND customer_id IS NOT NULL AND invoice_ts < ? GROUP BY 1)
        SELECT s.customer_id, min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS orders, sum(amt) AS gross,
               coalesce(any_value(r.ret), 0) AS returns
        FROM s LEFT JOIN r USING (customer_id) GROUP BY s.customer_id""", [snapshot, snapshot]).df()
    df["net"] = df["gross"] - df["returns"]
    df["recency_days"] = (snapshot - df["last_ts"]).dt.days
    df["tenure_days"] = (df["last_ts"] - df["first_ts"]).dt.days
    rk = lambda s: s.rank(method="first")
    df["R"] = pd.qcut(rk(-df["recency_days"]), 5, labels=False) + 1  # recency nho = diem cao
    df["F"] = pd.qcut(rk(df["orders"]), 5, labels=False) + 1
    df["M"] = pd.qcut(rk(df["gross"]), 5, labels=False) + 1
    df["segment"] = df.apply(_segment, axis=1)
    return df


def _segment(r):
    R, F, M = r["R"], r["F"], r["M"]
    if R >= 4 and F >= 4 and M >= 4:
        return "Champions"
    if F >= 4 and R >= 3:
        return "Loyal"
    if R <= 2 and F >= 3:
        return "At risk"
    if R == 5 and F == 1:
        return "New"
    if R >= 4 and F in (2, 3):
        return "Promising"
    if R <= 2 and F <= 2:
        return "Hibernating"
    return "Needs attention"


def cohorts(con, max_k=12):
    d = con.execute("""
        WITH m AS (SELECT customer_id, date_trunc('month', invoice_date) AS mon, sum(line_amount) AS amt
                   FROM clean_lines WHERE line_type='sale' AND customer_id IS NOT NULL GROUP BY 1, 2),
             f AS (SELECT customer_id, min(mon) AS cohort FROM m GROUP BY 1)
        SELECT f.cohort, m.mon, count(*) AS active, sum(m.amt) AS revenue
        FROM m JOIN f USING (customer_id) GROUP BY 1, 2""").df()
    d["k"] = (d["mon"].dt.year - d["cohort"].dt.year) * 12 + (d["mon"].dt.month - d["cohort"].dt.month)
    size = d[d["k"] == 0].set_index("cohort")["active"]
    d["size"] = d["cohort"].map(size)
    d["retention"] = d["active"] / d["size"]
    d["rev_per_customer"] = d["revenue"] / d["size"]
    return d[d["k"] <= max_k].sort_values(["cohort", "k"]).reset_index(drop=True)


def products(con):
    return con.execute("""
        SELECT c.stock_code, p.description,
               sum(CASE WHEN c.line_type='sale' THEN c.line_amount END) AS gross_sales,
               -sum(CASE WHEN c.line_type='cancellation' THEN c.line_amount END) AS returns,
               sum(CASE WHEN c.line_type='sale' THEN c.quantity END) AS units,
               count(DISTINCT CASE WHEN c.line_type='sale' THEN c.invoice END) AS orders,
               count(DISTINCT CASE WHEN c.line_type='sale' THEN c.customer_id END) AS customers
        FROM clean_lines c LEFT JOIN dim_product p USING (stock_code)
        WHERE c.line_type IN ('sale','cancellation') AND abs(c.line_amount) <= 50000 GROUP BY 1, 2 HAVING gross_sales > 0""").df()  # bo cac dong ngoai le > 50K (don 80.995 san pham 168K va lenh huy cua no)


def basket_pairs(con, top_n=300, min_count=150):
    return con.execute(f"""
        WITH inv AS (SELECT DISTINCT invoice, stock_code FROM clean_lines WHERE line_type='sale'),
             top AS (SELECT stock_code FROM inv GROUP BY 1 ORDER BY count(*) DESC LIMIT {top_n}),
             items AS (SELECT i.* FROM inv i JOIN top USING (stock_code)),
             n AS (SELECT count(DISTINCT invoice) AS total FROM inv),
             cnt AS (SELECT stock_code, count(*) AS c FROM inv GROUP BY 1),
             pr AS (SELECT a.stock_code AS a, b.stock_code AS b, count(*) AS together
                    FROM items a JOIN items b ON a.invoice = b.invoice AND a.stock_code < b.stock_code GROUP BY 1, 2 HAVING count(*) >= {min_count})
        SELECT pr.a, pa.description AS desc_a, pr.b, pb.description AS desc_b, pr.together,
               pr.together * 1.0 / n.total AS support,
               pr.together * 1.0 / ca.c AS confidence_a_to_b, pr.together * 1.0 / cb.c AS confidence_b_to_a,
               (pr.together * 1.0 / n.total) / ((ca.c * 1.0 / n.total) * (cb.c * 1.0 / n.total)) AS lift
        FROM pr, n JOIN cnt ca ON ca.stock_code = pr.a JOIN cnt cb ON cb.stock_code = pr.b
        JOIN dim_product pa ON pa.stock_code = pr.a JOIN dim_product pb ON pb.stock_code = pr.b
        ORDER BY lift DESC""").df()


def countries(con):
    return con.execute("""
        SELECT country,
               sum(CASE WHEN line_type='sale' AND invoice_ts < TIMESTAMP '2010-12-01' THEN line_amount END) AS y1,
               sum(CASE WHEN line_type='sale' AND invoice_ts >= TIMESTAMP '2010-12-01' THEN line_amount END) AS y2,
               sum(CASE WHEN line_type='sale' THEN line_amount END) AS gross_sales,
               count(DISTINCT CASE WHEN line_type='sale' THEN customer_id END) AS customers
        FROM clean_lines WHERE line_type='sale' GROUP BY 1 ORDER BY gross_sales DESC""").df()


def main():
    con = connect()
    t = ROOT / "outputs" / "tables"
    t.mkdir(parents=True, exist_ok=True)
    m = monthly(con); m.to_csv(t / "monthly.csv", index=False)
    w = weekly(con); w.to_csv(t / "weekly_net_revenue.csv", index=False)
    rfm = customers_rfm(con)
    seg = rfm.groupby("segment").agg(customers=("customer_id", "count"), net=("net", "sum"), avg_orders=("orders", "mean"), avg_recency_days=("recency_days", "mean")).reset_index()
    seg["share_of_net"] = seg["net"] / seg["net"].sum(); seg.sort_values("net", ascending=False).to_csv(t / "rfm_segments.csv", index=False)
    c = cohorts(con); c.to_csv(t / "cohorts.csv", index=False)
    p = products(con).sort_values("gross_sales", ascending=False); p.head(200).to_csv(t / "top_products.csv", index=False)
    b = basket_pairs(con); b.head(100).to_csv(t / "basket_pairs.csv", index=False)
    co = countries(con); co.to_csv(t / "countries.csv", index=False)
    print("monthly rows", len(m), "weekly rows", len(w), "customers w/ id", len(rfm), "cohort rows", len(c), "products", len(p), "pairs", len(b), "countries", len(co))
    print(seg.sort_values("net", ascending=False).round(2).to_string(index=False))
    print(m.tail(3).round(0).to_string(index=False))
    print(b.head(5)[["desc_a", "desc_b", "together", "lift"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

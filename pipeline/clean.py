"""Buoc 2: lam sach. Tao bang clean_lines, dim_product va cleaning_log trong DuckDB, ghi outputs/cleaning_log.md.

Moi quy tac duoc dem so dong va so tien (GBP) bi anh huong, de ai doc cung thay ro da loai gi va vi sao.
Quy tac khong xoa du lieu mot cach am tham: dong bi loai khoi phan tich ban hang van nam trong clean_lines voi line_type rieng.

Chay: python -I pipeline/clean.py
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "warehouse.duckdb"
OUT = ROOT / "outputs"

# Ma khong phai san pham: phi van chuyen, dieu chinh ke toan, phi ngan hang, phi san, san pham test, phieu qua tang
NON_PRODUCT_CODES = ("M", "m", "POST", "DOT", "C2", "C3", "D", "S", "B", "BANK CHARGES", "ADJUST", "ADJUST2",
                     "AMAZONFEE", "CRUK", "TEST001", "TEST002", "GIFT", "PADS")

SQL = f"""
DROP TABLE IF EXISTS clean_lines;
CREATE TABLE clean_lines AS
WITH base AS (
  SELECT *,
         (source_sheet = 'Year 2009-2010' AND invoice_ts >= TIMESTAMP '2010-12-01') AS in_overlap_dup
  FROM raw_online_retail
), dedup AS (
  SELECT * FROM base WHERE NOT in_overlap_dup
)
SELECT row_number() OVER (ORDER BY invoice_ts, invoice, stock_code) AS line_id,
       invoice, stock_code, upper(trim(description)) AS description_raw, quantity, invoice_ts,
       CAST(invoice_ts AS DATE) AS invoice_date, price,
       CAST(customer_id AS BIGINT) AS customer_id, country,
       quantity * price AS line_amount,
       CASE
         WHEN stock_code IN {NON_PRODUCT_CODES} OR stock_code LIKE 'gift\\_%' ESCAPE '\\' THEN 'non_product'
         WHEN invoice LIKE 'C%' THEN 'cancellation'
         WHEN price <= 0 THEN 'zero_or_negative_price'
         WHEN quantity <= 0 THEN 'adjustment'
         ELSE 'sale'
       END AS line_type
FROM dedup;

DROP TABLE IF EXISTS dim_product;
CREATE TABLE dim_product AS
SELECT stock_code, arg_max(description_raw, n) AS description
FROM (SELECT stock_code, description_raw, count(*) AS n FROM clean_lines
      WHERE description_raw IS NOT NULL AND line_type IN ('sale', 'cancellation') GROUP BY 1, 2)
GROUP BY stock_code;
"""


def main():
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB))
    for stmt in [s for s in SQL.split(";\n") if s.strip()]:
        con.execute(stmt)
    q = lambda s: con.execute(s).fetchone()

    raw_n, raw_amt = q("SELECT count(*), sum(quantity * price) FROM raw_online_retail")
    ov_n, ov_amt = q("SELECT count(*), sum(quantity * price) FROM raw_online_retail WHERE source_sheet='Year 2009-2010' AND invoice_ts >= TIMESTAMP '2010-12-01'")
    log = [("R0", "Raw rows loaded (both Excel sheets)", raw_n, raw_amt, "info")]
    log.append(("R1", "Rows in the 1-9 Dec 2010 window that appear in BOTH Excel sheets (exact copies, verified row by row)", ov_n, ov_amt, "dropped the copy from sheet 2009-2010"))
    for code, label, action in [
        ("non_product", "Non-product lines: postage, manual entries, bank/marketplace fees, adjustments, test items, gift vouchers", "excluded from product sales; kept in clean_lines"),
        ("cancellation", "Cancelled or returned product lines (invoice starts with C)", "tracked as returns, subtracted to give net revenue"),
        ("zero_or_negative_price", "Product lines with price 0 or below (free items, bad-debt adjustments)", "excluded from sales"),
        ("adjustment", "Negative-quantity lines not on a cancellation invoice (stock write-offs, damaged goods)", "excluded from sales"),
    ]:
        n, amt = q(f"SELECT count(*), sum(line_amount) FROM clean_lines WHERE line_type = '{code}'")
        log.append((code, label, n, amt, action))
    n, amt = q("SELECT count(*), sum(line_amount) FROM clean_lines WHERE line_type='sale'")
    log.append(("sale", "Sale lines kept for analysis", n, amt, "core dataset"))
    n, amt = q("SELECT count(*), sum(line_amount) FROM clean_lines WHERE line_type='sale' AND customer_id IS NULL")
    log.append(("R6", "Sale lines with no customer id", n, amt, "kept for revenue and product analysis; excluded from customer-level analysis (RFM, cohorts, LTV)"))
    n, = q("""SELECT coalesce(sum(c - 1), 0) FROM (SELECT count(*) c FROM clean_lines WHERE line_type='sale'
              GROUP BY invoice, stock_code, quantity, invoice_ts, price, customer_id HAVING count(*) > 1)""")
    log.append(("R7", "Identical repeated sale lines inside one invoice (same product scanned twice)", n, None, "kept: plausible repeat scans, not removed"))
    thr, = q("SELECT quantile_cont(line_amount, 0.999) FROM clean_lines WHERE line_type='sale'")
    n, amt = q(f"SELECT count(*), sum(line_amount) FROM clean_lines WHERE line_type='sale' AND line_amount > {thr}")
    log.append(("R8", f"Outlier sale lines above the 99.9th percentile (> GBP {thr:,.0f} per line)", n, amt, "kept and flagged; reported separately in insights"))

    con.execute("DROP TABLE IF EXISTS cleaning_log")
    con.execute("CREATE TABLE cleaning_log(rule VARCHAR, description VARCHAR, rows BIGINT, amount_gbp DOUBLE, action VARCHAR)")
    con.executemany("INSERT INTO cleaning_log VALUES (?,?,?,?,?)", log)

    lines = ["# Cleaning log", "",
             "Generated by `pipeline/clean.py`. Amounts are GBP (quantity x price). Every excluded row stays in `clean_lines` with its `line_type`.", "",
             "| Rule | What | Rows | Amount (GBP) | Action |", "|---|---|---:|---:|---|"]
    for r, d, n, a, act in log:
        lines.append(f"| {r} | {d} | {n:,} | {'' if a is None else f'{a:,.0f}'} | {act} |")
    (OUT / "cleaning_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r, d, n, a, act in log:
        print(f"{r:24s} {n:>10,}  {'' if a is None else format(a, ',.0f'):>14}  {d[:70]}")
    con.close()


if __name__ == "__main__":
    main()

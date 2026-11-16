"""Buoc 1: doc file Excel goc UCI Online Retail II (2 sheet) va nap nguyen ban vao DuckDB (bang raw_online_retail).

Du lieu: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5CG6D. Giay phep CC BY 4.0.
Du lieu la giao dich that cua mot nha ban le truc tuyen o Anh (12/2009 den 12/2011); khach hang chi co ma so da an danh.
File goc khong duoc dua vao git (xem .gitignore); chay `python -I pipeline/download.py` de lay lai.

Chay: python -I pipeline/ingest.py
"""
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "warehouse.duckdb"


def main():
    xlsx = RAW / "online_retail_II.xlsx"
    if not xlsx.exists():
        z = RAW / "online_retail_II.zip"
        if not z.exists():
            sys.exit("Thieu du lieu. Chay: python -I pipeline/download.py")
        with zipfile.ZipFile(z) as zf:
            assert zf.namelist() == ["online_retail_II.xlsx"], zf.namelist()  # chi giai nen dung mot file da biet
            zf.extract("online_retail_II.xlsx", RAW)
    t0 = time.time()
    frames = []
    for sheet in ("Year 2009-2010", "Year 2010-2011"):
        df = pd.read_excel(xlsx, sheet_name=sheet, dtype={"Invoice": str, "StockCode": str})
        df["source_sheet"] = sheet
        frames.append(df)
        print(f"  {sheet}: {len(df):,} dong ({time.time() - t0:.0f}s)")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"Invoice": "invoice", "StockCode": "stock_code", "Description": "description", "Quantity": "quantity",
                              "InvoiceDate": "invoice_ts", "Price": "price", "Customer ID": "customer_id", "Country": "country"})
    con = duckdb.connect(str(DB))
    con.execute("DROP TABLE IF EXISTS raw_online_retail")
    con.register("raw_df", raw)
    con.execute("CREATE TABLE raw_online_retail AS SELECT * FROM raw_df")
    n = con.execute("SELECT COUNT(*) FROM raw_online_retail").fetchone()[0]
    print(f"raw_online_retail: {n:,} dong")
    con.close()


if __name__ == "__main__":
    main()

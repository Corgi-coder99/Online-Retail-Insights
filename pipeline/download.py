"""Tai du lieu UCI Online Retail II vao data/raw (khong dua vao git).

Nguon: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0). Dung luong khoang 43,5 MB.
Kiem tra SHA256 de dam bao dung file da dung khi viet du an.
Chay: python -I pipeline/download.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail_II.zip"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        print("Dang tai", URL)
        urllib.request.urlretrieve(URL, OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest != SHA256:
        sys.exit(f"SHA256 khong khop: {digest}. Khong dung file nay.")
    print(f"OK {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), SHA256 khop")


if __name__ == "__main__":
    main()

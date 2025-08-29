# scrape_company_list.py
import os
import time
import pandas as pd
import requests

# ---------- SETTINGS ----------
SAVE_DIR = r"D:\Data Science\DSA Codes\Python-Basics\Project-Stock-App"
os.makedirs(SAVE_DIR, exist_ok=True)

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATAHUB_SP500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

NSE_HOME = "https://www.nseindia.com"
NSE_NIFTY500 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
WIKI_NIFTY500 = "https://en.wikipedia.org/wiki/NIFTY_500"

HEADERS_COMMON = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_with_retries(url, session=None, headers=None, retries=5, backoff=1.6, expect="text"):
    """Generic fetch with headers, retries, and backoff."""
    if session is None:
        session = requests.Session()
    if headers is None:
        headers = HEADERS_COMMON

    last_exc = None
    for i in range(retries):
        try:
            resp = session.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.text if expect == "text" else resp.content
            # Some sites return 403/429—sleep & retry
        except Exception as e:
            last_exc = e
        time.sleep(backoff ** (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")


def load_sp500() -> pd.DataFrame:
    """S&P 500 via Wikipedia; fallback to DataHub CSV."""
    print("→ Fetching S&P 500 …")
    sess = requests.Session()
    # Try Wikipedia first
    try:
        html = fetch_with_retries(WIKI_SP500, session=sess, headers=HEADERS_COMMON, expect="text")
        tables = pd.read_html(html)
        # Wikipedia first table is constituents
        df = tables[0]
        # Standardize
        if "Symbol" in df.columns and "Security" in df.columns:
            out = df[["Symbol", "Security"]].copy()
            out.columns = ["Symbol", "Company"]
        else:
            # fallback column detection
            sym = [c for c in df.columns if "Symbol" in str(c)]
            nam = [c for c in df.columns if "Security" in str(c) or "Company" in str(c)]
            out = df[[sym[0], nam[0]]].copy()
            out.columns = ["Symbol", "Company"]
        out["Market"] = "US"
        print(f"   ✓ S&P 500 (Wikipedia): {len(out)} rows")
        return out
    except Exception as e:
        print(f"   ! Wikipedia failed ({e}). Trying DataHub fallback…")

    # Fallback DataHub
    try:
        csv_bytes = fetch_with_retries(DATAHUB_SP500, session=sess, headers=HEADERS_COMMON, expect="bytes")
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
        # DataHub columns: Symbol, Name
        out = df.rename(columns={"Name": "Company"})[["Symbol", "Company"]].copy()
        out["Market"] = "US"
        print(f"   ✓ S&P 500 (DataHub): {len(out)} rows")
        return out
    except Exception as e2:
        raise RuntimeError(f"Failed all S&P 500 sources: {e2}")


def load_nifty500() -> pd.DataFrame:
    """NIFTY 500 via NSE (needs cookies/headers); fallback to Wikipedia."""
    print("→ Fetching NIFTY 500 …")
    sess = requests.Session()

    # Seed cookies by visiting NSE home first
    try:
        _ = fetch_with_retries(NSE_HOME, session=sess, headers=HEADERS_COMMON, expect="text")
        nse_headers = {
            **HEADERS_COMMON,
            "Referer": "https://www.nseindia.com/",
            "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
        }
        csv_bytes = fetch_with_retries(NSE_NIFTY500, session=sess, headers=nse_headers, expect="bytes")
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
        # Common NSE schema: Symbol, Company Name
        if "Company Name" in df.columns:
            out = df.rename(columns={"Company Name": "Company"})[["Symbol", "Company"]].copy()
        else:
            # Try best-effort detection
            sym = [c for c in df.columns if "Symbol" in str(c)]
            nam = [c for c in df.columns if "Company" in str(c)]
            out = df[[sym[0], nam[0]]].copy()
            out.columns = ["Symbol", "Company"]
        out["Market"] = "India"
        print(f"   ✓ NIFTY 500 (NSE): {len(out)} rows")
        return out
    except Exception as e:
        print(f"   ! NSE CSV failed ({e}). Trying Wikipedia fallback…")

    # Fallback Wikipedia
    try:
        html = fetch_with_retries(WIKI_NIFTY500, session=sess, headers=HEADERS_COMMON, expect="text")
        tables = pd.read_html(html)
        # Find a table that has 'Symbol' and 'Company'
        candidate = None
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("symbol" in c for c in cols) and (
                any("company" in c for c in cols) or any("security" in c for c in cols)
            ):
                candidate = t
                break
        if candidate is None:
            raise RuntimeError("No matching table found on Wikipedia NIFTY 500")

        # standardize
        df = candidate.copy()
        # pick first matching name-like column
        name_col = None
        for c in df.columns:
            lc = str(c).lower()
            if "company" in lc or "security" in lc or "name" in lc:
                name_col = c
                break
        sym_col = [c for c in df.columns if "Symbol" in str(c) or str(c).lower().strip() == "symbol"][0]
        out = df[[sym_col, name_col]].copy()
        out.columns = ["Symbol", "Company"]
        out["Market"] = "India"
        print(f"   ✓ NIFTY 500 (Wikipedia): {len(out)} rows")
        return out
    except Exception as e2:
        raise RuntimeError(f"Failed all NIFTY 500 sources: {e2}")


def add_yahoo_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Create a Yahoo Finance-compatible symbol column."""
    df = df.copy()
    def to_yf(row):
        sym = str(row["Symbol"]).strip()
        if row["Market"] == "US":
            # Yahoo uses '-' instead of '.' for class shares, e.g., BRK.B -> BRK-B
            return sym.replace(".", "-")
        if row["Market"] == "India":
            # Yahoo NSE suffix
            return f"{sym}.NS"
        return sym

    df["YahooSymbol"] = df.apply(to_yf, axis=1)
    return df


def dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["Symbol", "Market"]).copy()
    return df.sort_values(["Market", "Symbol"]).reset_index(drop=True)


def main():
    sp500 = load_sp500()
    nifty500 = load_nifty500()

    sp500 = dedupe_sort(sp500)
    nifty500 = dedupe_sort(nifty500)

    all_stocks = pd.concat([sp500, nifty500], ignore_index=True)
    all_stocks = dedupe_sort(all_stocks)

    # Add Yahoo tickers
    all_stocks_yahoo = add_yahoo_symbol(all_stocks)

    # Save
    sp500.to_csv(os.path.join(SAVE_DIR, "sp500.csv"), index=False, encoding="utf-8-sig")
    nifty500.to_csv(os.path.join(SAVE_DIR, "nifty500.csv"), index=False, encoding="utf-8-sig")
    all_stocks.to_csv(os.path.join(SAVE_DIR, "all_stocks.csv"), index=False, encoding="utf-8-sig")
    all_stocks_yahoo.to_csv(os.path.join(SAVE_DIR, "all_stocks_yahoo.csv"), index=False, encoding="utf-8-sig")

    print("—" * 60)
    print("Saved files:")
    for fn in ["sp500.csv", "nifty500.csv", "all_stocks.csv", "all_stocks_yahoo.csv"]:
        p = os.path.join(SAVE_DIR, fn)
        print(f"  ✓ {p} ({pd.read_csv(p).shape[0]} rows)")
    print("Done ✅")


if __name__ == "__main__":
    main()

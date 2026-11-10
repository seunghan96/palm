#!/usr/bin/env python3
"""Rebuild the paper's panel from FNSPID and the shipped list of article-days.

The paper scores 252,000 article-days: 25,200 per calendar year from 2014 to
2023, drawn from the wider panel that build_panel.py produces. FNSPID carries
no article identifier, so data/panel_ids.parquet names each row by its UTC
timestamp, its ticker and the SHA-1 of the text the pipeline composed for it
(the headline, followed by FNSPID's own summary when one exists). Everything
else the panel needs, the trading day, the publisher and the next-session
return, is already in that file. The text is not redistributed here; it is read
from the FNSPID release, which is CC BY-NC 4.0.

    python3 palm/restore_panel.py --news data/nasdaq_exteral_data.csv

writes data/panel.parquet with the same rows, in the same order, under the same
aid as the paper, and refuses to write anything if a single row is missing.
"""
import argparse, hashlib, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, PANEL, PANEL_IDS
from build_panel import collect_news

KEY = ["ts", "symbol", "text_sha1"]


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default=f"{DATA}/nasdaq_exteral_data.csv")
    ap.add_argument("--ids", default=PANEL_IDS)
    ap.add_argument("--out", default=PANEL)
    a = ap.parse_args()

    ids = pd.read_parquet(a.ids)
    years = ids["date"].dt.year
    print(f"article-days to restore: {len(ids):,} ({years.min()}-{years.max()})", flush=True)

    print("== reading FNSPID ==", flush=True)
    raw = collect_news(a.news, int(years.min()), int(years.max()))
    raw = raw[raw["symbol"].isin(set(ids["symbol"]))].copy()
    print(f"candidate rows with a panel ticker: {len(raw):,}", flush=True)
    raw["text_sha1"] = raw["text"].map(sha1)
    raw = raw.drop_duplicates(KEY)[KEY + ["text"]]

    out = ids.merge(raw, on=KEY, how="left")
    missing = out["text"].isna()
    if missing.any():
        print(f"{int(missing.sum()):,} rows found no text. Nothing written.", file=sys.stderr)
        print(out.loc[missing, ["aid", "ts", "symbol"]].head(10).to_string(index=False), file=sys.stderr)
        sys.exit(1)
    assert len(out) == len(ids) and (out["aid"].values == ids["aid"].values).all()

    out = out[["date", "symbol", "ts", "text", "publisher", "fwd_ret", "aid"]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(out):,} rows, {out['symbol'].nunique():,} tickers, "
          f"{out['date'].min().date()} to {out['date'].max().date()}")


if __name__ == "__main__":
    main()

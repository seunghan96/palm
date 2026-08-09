#!/usr/bin/env python3
"""
Build the article-day panel from FNSPID: (date, symbol, text) + forward return.

Time alignment (this is the part that must not cheat)
-----------------------------------------------------
FNSPID `Date` carries a real UTC timestamp, verified on the raw file as
"2023-12-16 23:00:00 UTC" rather than a bare midnight, so we can use it.

US equity close = 16:00 ET = 20:00 UTC (EDT) / 21:00 UTC (EST). We take the
CONSERVATIVE cut at 20:00 UTC: an article is only credited to trading day d if
its timestamp is strictly before 20:00 UTC on d. Otherwise it moves to the next
trading day. This can only ever delay information, never advance it.

Target: r_{d->d+1} = adjclose(d+1)/adjclose(d) - 1, where d is the assignment
day. So the signal is formed from news known by the close of d, and the return
is realised over the following session. No same-bar leakage.
"""
import argparse, io, json, os, sys, time, zipfile
import numpy as np
import pandas as pd

from paths import DATA, RESULTS
OUT = RESULTS
CUT_HOUR_UTC = 20


def load_prices(zip_path, symbols=None):
    """Return dict symbol -> DataFrame(date, adjclose) sorted ascending."""
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist()
             if n.startswith("full_history/") and n.endswith(".csv")]
    out = {}
    for i, n in enumerate(names):
        sym = os.path.basename(n)[:-4].upper()
        if symbols is not None and sym not in symbols:
            continue
        try:
            df = pd.read_csv(z.open(n), usecols=["date", "adj close"])
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        df = df.rename(columns={"adj close": "adjclose"})
        df = df[df["adjclose"] > 0]
        if len(df) > 100:
            out[sym] = df
    return out


def collect_news(csv_path, year_min, year_max, chunksize=200_000, max_rows=None):
    use = ["Date", "Article_title", "Stock_symbol", "Lsa_summary", "Publisher"]
    keep = []
    n_seen = 0
    t0 = time.time()
    it = pd.read_csv(csv_path, chunksize=chunksize, usecols=use,
                     low_memory=False, on_bad_lines="skip")
    ci = -1
    while True:
        ci += 1
        try:
            ch = next(it)
        except StopIteration:
            break
        except Exception as e:          # truncated file (partial download)
            print(f"  reader stopped at chunk {ci}: {type(e).__name__}", flush=True)
            break
        n_seen += len(ch)
        ch = ch[ch["Stock_symbol"].notna()]
        if len(ch) == 0:
            continue
        dt = pd.to_datetime(ch["Date"], errors="coerce", utc=True, format="mixed")
        ch = ch.assign(ts=dt).dropna(subset=["ts"])
        ch = ch[(ch["ts"].dt.year >= year_min) & (ch["ts"].dt.year <= year_max)]
        if len(ch) == 0:
            continue
        # text: prefer the LSA summary (FNSPID's own, ~3 sentences), else title
        txt = ch["Lsa_summary"].fillna("").astype(str).str.strip()
        title = ch["Article_title"].fillna("").astype(str).str.strip()
        text = np.where(txt.str.len() >= 40, title + ". " + txt, title)
        ch = ch.assign(text=text)
        ch = ch[ch["text"].str.len() >= 20]
        keep.append(ch[["ts", "Stock_symbol", "text", "Publisher"]])
        if ci % 20 == 0:
            got = sum(len(k) for k in keep)
            print(f"  chunk {ci} seen={n_seen:,} kept={got:,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if max_rows and sum(len(k) for k in keep) >= max_rows:
            break
    df = pd.concat(keep, ignore_index=True)
    df = df.rename(columns={"Stock_symbol": "symbol", "Publisher": "publisher"})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df


def assign_trading_day(ts_utc, trading_days):
    """First trading day whose 20:00 UTC close is strictly after the timestamp."""
    eff = ts_utc.dt.normalize()
    late = ts_utc.dt.hour >= CUT_HOUR_UTC
    eff = eff + pd.to_timedelta(late.astype(int), unit="D")
    eff = eff.dt.tz_localize(None)
    idx = np.searchsorted(trading_days, eff.values, side="left")
    ok = idx < len(trading_days)
    res = np.full(len(eff), np.datetime64("NaT"), dtype="datetime64[ns]")
    res[ok] = trading_days[idx[ok]]
    return pd.to_datetime(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default=f"{DATA}/nasdaq_exteral_data.csv")
    ap.add_argument("--prices", default=f"{DATA}/full_history.zip")
    ap.add_argument("--year-min", type=int, default=2014)
    ap.add_argument("--year-max", type=int, default=2023)
    ap.add_argument("--top-symbols", type=int, default=500)
    ap.add_argument("--max-per-symbol-year", type=int, default=60)
    ap.add_argument("--max-news-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{DATA}/panel.parquet")
    a = ap.parse_args()

    print("== collecting news ==", flush=True)
    news = collect_news(a.news, a.year_min, a.year_max, max_rows=a.max_news_rows)
    print(f"news rows with symbol+text in range: {len(news):,}", flush=True)

    counts = news["symbol"].value_counts()
    top = set(counts.head(a.top_symbols).index)
    news = news[news["symbol"].isin(top)]
    print(f"after top-{a.top_symbols} symbols: {len(news):,}", flush=True)

    print("== loading prices ==", flush=True)
    prices = load_prices(a.prices, symbols=top)
    print(f"price series loaded: {len(prices)}", flush=True)
    news = news[news["symbol"].isin(prices.keys())]
    print(f"after price-match: {len(news):,}", flush=True)

    # balanced subsample: cap articles per (symbol, year) so no symbol dominates
    rng = np.random.RandomState(a.seed)
    news["year"] = news["ts"].dt.year
    news = (news.groupby(["symbol", "year"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), a.max_per_symbol_year),
                                          random_state=a.seed)))
    print(f"after per-(symbol,year) cap: {len(news):,}", flush=True)

    rows = []
    for sym, g in news.groupby("symbol"):
        p = prices[sym]
        td = p["date"].values
        d = assign_trading_day(g["ts"], td)
        gg = g.assign(date=d).dropna(subset=["date"])
        m = p.reset_index(drop=True)
        pos = pd.Series(m.index.values, index=m["date"].values)
        i0 = gg["date"].map(pos)
        i1 = i0 + 1
        valid = i1 < len(m)
        gg = gg[valid.values]
        i0 = i0[valid.values].astype(int)
        i1 = i1[valid.values].astype(int)
        r = m["adjclose"].values[i1] / m["adjclose"].values[i0] - 1.0
        gg = gg.assign(fwd_ret=r)
        rows.append(gg[["date", "symbol", "ts", "text", "publisher", "fwd_ret"]])

    panel = pd.concat(rows, ignore_index=True)
    panel = panel[np.isfinite(panel["fwd_ret"])]
    panel = panel[panel["fwd_ret"].abs() < 0.5]      # drop split/data artifacts
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["aid"] = np.arange(len(panel), dtype=np.int64)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    panel.to_parquet(a.out, index=False)

    summ = {
        "rows": len(panel),
        "symbols": panel["symbol"].nunique(),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "fwd_ret_mean": float(panel["fwd_ret"].mean()),
        "fwd_ret_sd": float(panel["fwd_ret"].std()),
        "text_len_median": int(panel["text"].str.len().median()),
        "by_year": panel.groupby(panel["date"].dt.year).size().to_dict(),
        "intraday_time_share": float((panel["ts"].dt.hour != 0).mean()),
    }
    with open(f"{OUT}/panel_summary.json", "w") as f:
        json.dump(summ, f, indent=2, default=str)
    print("PANEL " + json.dumps(summ, default=str), flush=True)


if __name__ == "__main__":
    main()

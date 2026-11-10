#!/usr/bin/env python3
"""Backtest specification and point-in-time audit.

Everything a reader needs to check the panel, computed rather than asserted:
  - timestamp granularity: the share of articles stamped exactly at midnight
    UTC, which can only be date-precise, and how the 20:00 UTC rule treats them
  - assignment: no article is credited to a trading day after whose close it
    was published
  - cross-section: articles, distinct tickers, and multi-article tickers per day
  - duplicates: identical texts within a day (syndication) and across the
    decision date (a fit-window text reappearing in the evaluation window)
  - fit windows: every adapted panel on disk records the latest month its
    adapter read (fit_hi); all must sit before the decision date

    python3 palm/audit.py      -> results/audit.json"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PANEL, BIG, RESULTS

p = pd.read_parquet(PANEL)
p["t"] = p.date.dt.year + p.date.dt.month / 12.0
R = {}

midnight = (p.ts.dt.hour == 0) & (p.ts.dt.minute == 0) & (p.ts.dt.second == 0)
R["n_articles"] = len(p)
R["share_midnight_stamp"] = float(midnight.mean())
# a date-only stamp is read as 00:00 UTC, i.e. before the close of that date.
same_day = p.ts.dt.tz_convert(None).dt.normalize() == p.date
R["share_credited_same_day"] = float(same_day.mean())
late = p.ts.dt.tz_convert(None) >= p.date + pd.Timedelta(hours=20)
R["n_after_close_violations"] = int(late.sum())

g = p.groupby("date")
R["articles_per_day"] = dict(median=float(g.size().median()), mean=float(g.size().mean()),
                             min=int(g.size().min()), max=int(g.size().max()))
R["tickers_per_day"] = dict(median=float(g.symbol.nunique().median()),
                            mean=float(g.symbol.nunique().mean()))
multi = p.groupby(["date", "symbol"]).size()
R["share_ticker_days_with_multiple_articles"] = float((multi > 1).mean())
R["n_dates"] = int(p.date.nunique())
bd = pd.bdate_range(p.date.min(), p.date.max())
R["business_days_without_news"] = int(len(set(bd) - set(p.date)))
R["n_publishers"] = int(p.publisher.nunique())
R["n_tickers"] = int(p.symbol.nunique())

key = p.text.str.strip().str.lower()
R["share_exact_duplicate_text_same_day"] = float(p.assign(k=key).duplicated(["date", "k"]).mean())
fit, ev = key[p.t < 2021.0], key[p.t >= 2021.0]
R["eval_texts_seen_verbatim_before_decision"] = int(ev.isin(set(fit)).sum())
R["share_eval_texts_seen_verbatim_before_decision"] = float(ev.isin(set(fit)).mean())

ev_rows = p[p.t >= 2021.0]
R["eval_window"] = dict(rows=len(ev_rows), dates=int(ev_rows.date.nunique()),
                        first=str(ev_rows.date.min().date()), last=str(ev_rows.date.max().date()))
R["fit_side"] = dict(rows=int((p.t < 2021.0).sum()), last=str(p[p.t < 2021.0].date.max().date()))

viol, nfiles = [], 0
for f in glob.glob(f"{BIG}/*/*.parquet"):
    d = pd.read_parquet(f)
    if "fit_hi" not in d.columns or d["mode"].iloc[0] != "adapt" or np.isnan(d.final_loss.iloc[0]):
        continue
    nfiles += 1
    hi, evf = float(d.fit_hi.iloc[0]), float(d.eval_from.iloc[0])
    if os.path.basename(os.path.dirname(f)).startswith("oracle"):
        continue                       # reads past the date on purpose, reported as a bound
    if not np.isnan(hi) and hi >= evf:
        viol.append(os.path.relpath(f, BIG))
R["adapted_panels_checked"] = nfiles
R["adapted_panels_reading_eval_window"] = viol

json.dump(R, open(f"{RESULTS}/audit.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

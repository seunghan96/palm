#!/usr/bin/env python3
"""Dependence-aware inference on the cached daily series.

Every estimate here is a function of the cached daily series
(results/cache/<arm>/<model>.parquet, written by daily.py), and every interval
comes from one resampling scheme, fixed before it was run:

  - dates are resampled in calendar-month blocks, which keeps the serial and
    cross-checkpoint dependence within a month (all checkpoints share a day)
  - checkpoints are resampled with replacement *within* each suite, which
    keeps the suite composition fixed and lets checkpoints of one suite be as
    correlated as they are
  - both at once, B = 2,000 draws (NBOOT), percentile intervals

Quantities:
  paired(a, b)        mean over checkpoints of the per-checkpoint mean daily
                      difference (a - b) in IC, return and Sharpe
  slope(arm)          IC (or Sharpe) on staleness with suite fixed effects,
                      with a TOST non-inferiority read against a margin
  net_sharpe(arm, c)  Sharpe after a one-way cost of c bp per unit turnover
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE

B = int(os.environ.get("NBOOT", "2000"))
EV = 2021.0                                   # test window opens (Dec 2020)
THREE = ("ChronoGPT", "ChronoGPT-Instruct", "DatedGPT")   # the pooled 44


def suite(m):
    if "chrono-gpt-instruct" in m: return "ChronoGPT-Instruct"
    if "chrono-gpt" in m: return "ChronoGPT"
    if "PIT-4B" in m: return "PIT-4B"
    if "datedgpt" in m: return "DatedGPT"
    return None


def cutoff(m):
    g = re.findall(r"(\d{6,8})", m)
    if g:
        d = g[-1]
        return int(d[:4]) + (int(d[4:6]) if len(d) >= 6 else 12) / 12.0
    g = re.findall(r"-(\d{4})", m)
    return int(g[-1]) + 1.0 if g else None


def tcode(dates):
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.asarray(d.year + d.month / 12.0)


def load(arm, ev=EV, ev_to=None, eligible=True):
    """{model: daily frame} for one arm, restricted to the window."""
    out = {}
    for f in glob.glob(f"{CACHE}/{arm}/*.parquet"):
        m = os.path.basename(f)[:-8]
        s, c = suite(m), cutoff(m)
        # eligible = cutoff strictly before the window opens. A vintage cut
        # at the end of December has cutoff == ev and has read the first month.
        if s is None or c is None or (eligible and c >= ev):
            continue
        d = pd.read_parquet(f)
        t = tcode(d.date)
        d = d[(t >= ev) & ((t < ev_to) if ev_to else True)]
        if len(d) >= 150:
            out[m] = d.set_index("date")
    return out


def _matrix(arms, col, ev=EV, ev_to=None, models=None, suites=THREE):
    """Aligned (checkpoint x date) matrices of `col` for each arm, on the
    checkpoints present in every arm and the dates each pair shares."""
    L = [load(a, ev, ev_to) for a in arms]
    ms = sorted(m for m in set.intersection(*[set(x) for x in L]) if suite(m) in suites)
    if models is not None:
        ms = [m for m in ms if m in models]
    dates = sorted(set().union(*[set(L[0][m].index) for m in ms]))
    M = []
    for x in L:
        M.append(np.vstack([x[m][col].reindex(dates).values for m in ms]))
    return ms, pd.to_datetime(dates), M


def _draws(ms, dates, rng):
    """One bootstrap draw: integer weights over dates (month blocks) and over
    checkpoints (within suite)."""
    ym = dates.year * 12 + dates.month
    months = np.unique(ym)
    pick = rng.choice(months, len(months), replace=True)
    cnt = pd.Series(pick).value_counts()
    wd = np.array([cnt.get(k, 0) for k in ym], dtype=float)
    wc = np.zeros(len(ms))
    su = np.array([suite(m) for m in ms])
    for s in np.unique(su):
        idx = np.where(su == s)[0]
        for i in rng.choice(idx, len(idx), replace=True):
            wc[i] += 1
    return wd, wc


def _pooled(D, wd, wc):
    """Weighted mean over checkpoints of each checkpoint's weighted mean over
    dates, ignoring dates a checkpoint did not score."""
    ok = ~np.isnan(D)
    num = np.where(ok, D, 0.0) @ wd
    den = ok.astype(float) @ wd
    per = np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)
    k = ~np.isnan(per) & (wc > 0)
    return float((per[k] * wc[k]).sum() / wc[k].sum())


def _sharpe_rows(R, wd):
    """Annualized Sharpe of each row under date weights (weights repeat days)."""
    out = []
    for r in R:
        ok = ~np.isnan(r) & (wd > 0)
        x, w = r[ok], wd[ok]
        mu = (x * w).sum() / w.sum()
        sd = np.sqrt(((x - mu) ** 2 * w).sum() / (w.sum() - 1))
        out.append(np.sqrt(252) * mu / sd if sd > 0 else np.nan)
    return np.array(out)


def paired(a, b, ev=EV, ev_to=None, models=None, seed=0, suites=THREE):
    """a minus b, paired within checkpoint and date."""
    rng = np.random.default_rng(seed)
    ms, dates, (Ia, Ib) = _matrix([a, b], "ic", ev, ev_to, models, suites)
    _, _, (Ra, Rb) = _matrix([a, b], "ret", ev, ev_to, models, suites)
    D = Ia - Ib
    one_d, one_c = np.ones(len(dates)), np.ones(len(ms))
    pt = dict(n_ckpt=len(ms), n_days=len(dates),
              ic_a=_pooled(Ia, one_d, one_c) * 1e3, ic_b=_pooled(Ib, one_d, one_c) * 1e3,
              d_ic=_pooled(D, one_d, one_c) * 1e3,
              d_ret=_pooled(Ra - Rb, one_d, one_c) * 252 * 100,
              d_sh=float(np.nanmean(_sharpe_rows(Ra, one_d) - _sharpe_rows(Rb, one_d))),
              wins_ic=float(np.mean(np.nanmean(Ia, 1) > np.nanmean(Ib, 1))),
              # share of checkpoints whose own return / Sharpe the arm raises
              wins_ret=float(np.mean(np.nanmean(Ra, 1) > np.nanmean(Rb, 1))),
              wins_sh=float(np.mean(_sharpe_rows(Ra, one_d) > _sharpe_rows(Rb, one_d))))
    bs = {"d_ic": [], "d_ret": [], "d_sh": []}
    for _ in range(B):
        wd, wc = _draws(ms, dates, rng)
        bs["d_ic"].append(_pooled(D, wd, wc) * 1e3)
        bs["d_ret"].append(_pooled(Ra - Rb, wd, wc) * 252 * 100)
        sh = _sharpe_rows(Ra, wd) - _sharpe_rows(Rb, wd)
        k = ~np.isnan(sh) & (wc > 0)
        bs["d_sh"].append(float((sh[k] * wc[k]).sum() / wc[k].sum()))
    for k, v in bs.items():
        v = np.array(v)
        pt[k + "_lo95"], pt[k + "_hi95"] = np.percentile(v, [2.5, 97.5])
        pt[k + "_p_le0"] = float(np.mean(v <= 0))       # one-sided bootstrap p
    pt["by_suite"] = {}
    su = np.array([suite(m) for m in ms])
    for s in np.unique(su):
        k = su == s
        pt["by_suite"][s] = dict(n=int(k.sum()),
                                 d_ic=_pooled(D[k], one_d, one_c[k]) * 1e3,
                                 wins=float(np.mean(np.nanmean(Ia[k], 1) > np.nanmean(Ib[k], 1))))
    return pt


def slope(arm, col="ic", ev=EV, margin=None, seed=0, suites=THREE):
    """Per-checkpoint mean (IC) or Sharpe on staleness, suite fixed effects.
    Returns the point slope, a 90% and a 95% interval, and the TOST verdict
    against a degradation margin (slope >= -margin)."""
    rng = np.random.default_rng(seed)
    ms, dates, (X,) = _matrix([arm], "ret" if col == "sharpe" else col, ev, suites=suites)
    st = np.array([ev - cutoff(m) for m in ms])
    su = np.array([suite(m) for m in ms])

    def fit(y, wc):
        k = ~np.isnan(y) & (wc > 0)
        Z = [st[k]] + [(su[k] == s).astype(float) for s in np.unique(su[k])]
        Z = np.column_stack(Z)
        W = np.sqrt(wc[k])
        beta = np.linalg.lstsq(Z * W[:, None], y[k] * W, rcond=None)[0]
        return beta[0]

    def ys(wd):
        if col == "sharpe":
            return _sharpe_rows(X, wd)
        ok = ~np.isnan(X)
        return (np.where(ok, X, 0) @ wd) / (ok.astype(float) @ wd)

    one_d, one_c = np.ones(len(dates)), np.ones(len(ms))
    b0 = fit(ys(one_d), one_c)
    bb = np.array([fit(ys(wd), wc) for wd, wc in (_draws(ms, dates, rng) for _ in range(B))])
    lo90, hi90 = np.percentile(bb, [5, 95])
    lo95, hi95 = np.percentile(bb, [2.5, 97.5])
    r = dict(arm=arm, col=col, n_ckpt=len(ms), slope=b0, lo90=lo90, hi90=hi90,
             lo95=lo95, hi95=hi95, p_two=float(2 * min(np.mean(bb <= 0), np.mean(bb >= 0))))
    if margin is not None:
        r["margin"] = margin
        r["noninferior"] = bool(lo90 > -margin)          # TOST, alpha = 0.05
        r["equivalent"] = bool(lo90 > -margin and hi90 < margin)
    return r


def net_sharpe(arm, bps=(0, 2, 5, 10), ev=EV, models=None):
    """Pooled Sharpe after one-way costs; also mean daily turnover."""
    L = load(arm, ev)
    rows = []
    for m, d in L.items():
        if models is not None and m not in models:
            continue
        to = d.turnover.fillna(d.turnover.mean())
        row = dict(model=m, suite=suite(m), turnover=float(to.mean()))
        for c in bps:
            r = d.ret - c * 1e-4 * to
            row[f"sh_{c}bp"] = float(np.sqrt(252) * r.mean() / r.std(ddof=1))
            row[f"ret_{c}bp"] = float(252 * r.mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(json.dumps(paired(sys.argv[1], sys.argv[2]), indent=1, default=float))

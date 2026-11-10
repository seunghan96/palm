#!/usr/bin/env python3
"""Every number and table the paper prints, from the cached daily series.

    NBOOT=2000 python3 palm/report.py
writes
    results/tables/*.tex      table bodies, one file per table in the paper
    results/numbers.tex       \\newcommand macros used in the prose
    results/report.json       everything, for checking
and prints the headline results.

Arms (results/big/<arm>, see configs/arms.txt):
    base_test     checkpoint as released, scored through adapt.py --no-train
    sel_ps        PALM at the per-backbone setting chosen on the validation window
    dom_ps        same budget, text from before the cutoff (domain control)
    shuf_ps / oracle_ps        the two bounds
    full_ft / sft_r2           the alternatives that also train
    base_hist / sel_ps_hist    same-firm retrieval without / with PALM
    mask_ps / seed{1,2,3}_ps   robustness
Primary target: next-day open-to-close return (cache suffix @oc1)."""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE, RESULTS, TABLES
import stats

A = "@oc1"
MAIN = os.environ.get("MAIN", "sel_ps")
BASE = "base_test"
R, M = {}, {}


def has(arm):
    return os.path.isdir(f"{CACHE}/{arm}") and len(os.listdir(f"{CACHE}/{arm}")) > 0


def sg(x, d=2):
    return f"{x:+.{d}f}"


def ci(r, k="d_ic"):
    return f"[{r[k + '_lo95']:+.2f}, {r[k + '_hi95']:+.2f}]"


def pct(x):
    return f"{100 * x:.0f}\\%"


def write(name, rows):
    open(f"{TABLES}/{name}.tex", "w").write("\n".join(rows) + "\n")


def paired(a, b, **kw):
    r = stats.paired(a, b, **kw)
    r.pop("by_suite", None)
    return r


if not (has(MAIN + A) and has(BASE + A)):
    sys.exit(f"need results/cache/{MAIN}{A} and results/cache/{BASE}{A}; run run/03_metrics.sh first")

# ------------------------------------------------------------- audit macros
if os.path.exists(f"{RESULTS}/audit.json"):
    au = json.load(open(f"{RESULTS}/audit.json"))
    M["ShareMidnight"] = f"{100 * au['share_midnight_stamp']:.1f}\\%"
    M["ArticlesPerDay"] = f"{au['articles_per_day']['median']:.0f}"
    M["TickersPerDay"] = f"{au['tickers_per_day']['median']:.0f}"
    M["ShareMultiArticle"] = f"{100 * au['share_ticker_days_with_multiple_articles']:.0f}\\%"
    M["ShareDupText"] = f"{100 * au['share_exact_duplicate_text_same_day']:.1f}\\%"

# ------------------------------------------------------------- alignment
# (A)-(D) are the four windows a score can be compared with
rows = []
for al, name in (("", "(A) Close $\\rightarrow$ Next close"), ("@on1", "(B) Close $\\rightarrow$ Next open"),
                 ("@oc1", "(C) Next open $\\rightarrow$ Next close"), ("@lag1", "(D) Next close $\\rightarrow$ Close after")):
    if not (has(MAIN + al) and has(BASE + al)):
        continue
    r = paired(MAIN + al, BASE + al)
    R[f"align{al or '@cc'}"] = r
    bf = lambda x, win: f"\\best{{{x:.2f}}}" if win else f"{x:.2f}"
    rows.append(f"{name} & {bf(r['ic_b'], r['ic_b'] > r['ic_a'])} & {bf(r['ic_a'], r['ic_a'] >= r['ic_b'])} & "
                f"\\best{{{sg(r['d_ic'])}}} {ci(r)} & \\best{{{pct(r['wins_ic'])}}} \\\\")
if rows:
    write("align", rows)

# ------------------------------------------------------------- staleness
# Each row states the largest yearly decay its 90% interval rules out
# (one-sided 5%), and whether a decay is detected (interval below zero).
rows, st = [], {}
for col, lab, sc in (("ic", "IC", 1e3), ("ret", "Return", 252 * 100), ("sharpe", "Sharpe", 1)):
    r = stats.slope(BASE + A, col)
    st[col] = r
    ruled = max(0.0, -r["lo90"]) * sc
    rows.append(f"{lab} & {sg(r['slope'] * sc, 3)} & [{r['lo90'] * sc:+.3f}, {r['hi90'] * sc:+.3f}] & "
                f"{ruled:.3f} & {'Yes' if r['hi90'] < 0 else 'No'} \\\\")
R["staleness"] = st
write("staleness", rows)

# ------------------------------------------------------------- main table
rows = []
for S, lab in ((("ChronoGPT",), "ChronoGPT"), (("ChronoGPT-Instruct",), "ChronoGPT-Instruct"),
               (("DatedGPT",), "DatedGPT"), (stats.THREE, "All backbones")):
    r = paired(MAIN + A, BASE + A, suites=S)
    if r["n_ckpt"] == 0:
        continue
    R[f"main_{lab}"] = r
    nb, na = stats.net_sharpe(BASE + A), stats.net_sharpe(MAIN + A)
    nb = nb[nb.suite.isin(S)]; na = na[na.suite.isin(S)]
    if lab == "All backbones":
        rows.append("\\midrule")
    pairs = [(r['ic_b'], r['ic_a']), (nb.ret_0bp.mean(), na.ret_0bp.mean()), (nb.sh_0bp.mean(), na.sh_0bp.mean())]
    bf = lambda x, win: f"\\best{{{x:.2f}}}" if win else f"{x:.2f}"
    cb = " & ".join(bf(b, b > a) for b, a in pairs)
    ca = " & ".join(bf(a, a >= b) for b, a in pairs)
    rows.append(f"{lab} & Base & {cb} & & \\\\")
    rows.append(f" & + PALM & {ca} & "
                f"\\best{{{sg(r['d_ic'])}}} {ci(r)} & \\best{{{pct(r['wins_ic'])}}} \\\\")
    if lab != "DatedGPT" and lab != "All backbones":
        rows.append("\\cmidrule(lr){1-7}")
write("main", rows)

# ------------------------------------------------------------- selection
if os.path.exists(f"{RESULTS}/selection.csv"):
    pool = pd.read_csv(f"{RESULTS}/selection.csv")
    best = pool.d_ic.max()
    pick = pool[pool.d_ic >= best - 0.05].sort_values(["rank", "steps"]).iloc[0]
    M["SelRank"], M["SelSteps"] = f"{int(pick['rank'])}", f"{int(pick.steps)}"
    M["SelLR"] = "$10^{-4}$" if abs(pick.lr - 1e-4) < 1e-12 else f"${pick.lr:.0e}$"
    M["NValCkpt"] = f"{int(pool.n_ckpt.max())}"
    rows = []
    for rk in sorted(pool["rank"].unique()):
        cells = []
        for s_ in (300, 600, 1200):
            x = pool[(pool["rank"] == rk) & (pool.steps == s_) & (np.isclose(pool.lr, 1e-4))]
            if len(x) == 0:
                cells.append("\\textemdash"); continue
            v = sg(x.d_ic.iloc[0])
            if x.arm.iloc[0] == pick.arm:
                v = f"\\best{{{v}}}"
            cells.append(v)
        rows.append(f"{rk} & " + " & ".join(cells) + " \\\\")
    write("select", rows)
    R["selection"] = pool.to_dict("records")

if os.path.exists(f"{RESULTS}/selection_by_suite.csv"):
    ps = pd.read_csv(f"{RESULTS}/selection_by_suite.csv")
    pick = {}
    for s_, g in ps.groupby("suite"):
        best = g.d_ic.max()
        pick[s_] = g[g.d_ic >= best - 0.05].sort_values(["rank", "steps"]).iloc[0]
    parts = [f"rank {int(pick[s_]['rank'])} with {int(pick[s_].steps)} steps for {s_}" for s_ in stats.THREE if s_ in pick]
    M["SelSetting"] = ", ".join(parts[:-1]) + ", and " + parts[-1] + ", all at a learning rate of $10^{-4}$"
    short = {"ChronoGPT": "Chrono", "ChronoGPT-Instruct": "Instr", "DatedGPT": "Dated"}
    for s_, g in ps.groupby("suite"):
        M[f"NVal{short[s_]}"] = str(int(g.n.max()))
    rows = []
    grid = ps[np.isclose(ps.lr, 1e-4)]
    for rk in sorted(grid["rank"].unique()):
        for st_ in (300, 600, 1200):
            cells = []
            for s_ in stats.THREE:
                x = grid[(grid.suite == s_) & (grid["rank"] == rk) & (grid.steps == st_)]
                if not len(x):
                    cells.append("\\textemdash"); continue
                v = f"{x.d_ic.iloc[0]:+.2f}"
                if s_ in pick and int(pick[s_]["rank"]) == rk and int(pick[s_].steps) == st_ and np.isclose(pick[s_].lr, 1e-4):
                    v = f"\\best{{{v}}}"
                cells.append(v)
            rows.append(f"{rk} & {st_} & " + " & ".join(cells) + " \\\\")
    oth = ps[~np.isclose(ps.lr, 1e-4)]
    if len(oth):
        rows.append("\\midrule")
        for (rk, lr_), g in oth.groupby(["rank", "lr"]):
            e = int(np.floor(np.log10(lr_))); c = lr_ / 10 ** e
            cells = [f"{g[g.suite == s_].d_ic.iloc[0]:+.2f}" if len(g[g.suite == s_]) else "\\textemdash" for s_ in stats.THREE]
            rows.append(f"{rk} & 600, ${c:.0f}\\times10^{{{e}}}$ & " + " & ".join(cells) + " \\\\")
    write("select_ps", rows)

# ------------------------------------------------------------- domain vs recency
if has("dom_ps" + A):
    dm = set(stats.load("dom_ps" + A))
    b = paired(MAIN + A, BASE + A, models=dm)
    d = paired("dom_ps" + A, BASE + A, models=dm)
    x = paired(MAIN + A, "dom_ps" + A, models=dm)
    R.update(e1_palm=b, e1_dom=d, e1_diff=x)
    M["NDomain"] = f"{b['n_ckpt']}"
    # Domain: the adapter reads financial news. Recency: it reads text the
    # checkpoint has never seen (after its cutoff, before the decision date).
    write("domain", [
        f"None (as released) & \\no & \\no & {b['ic_b']:.2f} & \\textemdash & \\textemdash \\\\",
        f"{{[A]}} Before the cutoff & \\yes & \\no & \\best{{{d['ic_a']:.2f}}} & \\best{{{sg(d['d_ic'])}}} {ci(d)} & \\best{{{pct(d['wins_ic'])}}} \\\\",
        f"{{[B]}} After the cutoff & \\yes & \\yes & \\best{{{b['ic_a']:.2f}}} & \\best{{{sg(b['d_ic'])}}} {ci(b)} & \\best{{{pct(b['wins_ic'])}}} \\\\"])

# ------------------------------------------------------------- bounds
rows = []
for arm, lab in ((MAIN, "PALM"), ("shuf_ps", "(a) Shuffled word order"), ("oracle_ps", "(b) Text past the date")):
    if not has(arm + A):
        continue
    r = paired(arm + A, BASE + A)
    R[f"bound_{arm}"] = r
    b_ = lambda v: f"\\best{{{v}}}"
    rows.append(f"{lab} & {b_(sg(r['d_ic']))} & {b_(pct(r['wins_ic']))} & {b_(sg(r['d_ret']))} & {b_(pct(r['wins_ret']))} & "
                f"{b_(sg(r['d_sh']))} & {b_(pct(r['wins_sh']))} \\\\")
write("bounds", rows)

# ------------------------------------------------------------- alternatives
alts = [(BASE, "Checkpoint as released", "\\textemdash"),
        ("sft_r2", "Supervised on past returns", "0.10\\%"),
        ("full_ft", "Full finetuning", "100\\%"),
        (MAIN, "PALM (Ours)", "0.05--0.45\\%"),
        ("base_hist", "Same-firm retrieval", "\\textemdash"),
        (MAIN + "_hist", "PALM + same-firm retrieval (Ours)", "0.05--0.45\\%")]
rows, n_alt = [], 0
for arm, lab, what in alts:
    if not has(arm + A):
        continue
    r = paired(arm + A, BASE + A)
    key = {MAIN: "palm", MAIN + "_hist": "palm_hist"}.get(arm, arm)
    R[f"alt_{key}"] = r
    n_alt = max(n_alt, r["n_ckpt"])
    _n = stats.net_sharpe(arm + A); shp = _n[_n.suite.isin(stats.THREE)].sh_0bp.mean()
    g = "\\textemdash" if arm == BASE else f"{sg(r['d_ic'])} {ci(r)}"
    w = "\\textemdash" if arm == BASE else pct(r["wins_ic"])
    lab2 = f"\\textbf{{{lab}}}" if "Ours" in lab else lab
    rows.append(f"{lab2} & {what} & {r['ic_a']:.2f} & {g} & {w} & {shp:.2f} \\\\")
if has(MAIN + "_hist" + A) and has("base_hist" + A):
    r = paired(MAIN + "_hist" + A, "base_hist" + A)
    R["alt_hist_increment"] = r
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{3}}{{l|}}{{PALM on top of retrieval}} & {sg(r['d_ic'])} {ci(r)} & {pct(r['wins_ic'])} & \\textemdash \\\\")
write("alternatives", rows)
M["NAltCkpt"] = f"{n_alt}"

# ------------------------------------------------------------- costs
rows = []
for arm, lab in ((BASE, "Base"), (MAIN, "+ PALM")):
    n = stats.net_sharpe(arm + A, bps=(0, 1, 2, 5))
    n = n[n.suite.isin(stats.THREE)]
    be = (n.ret_0bp / 100 / 252 / n.turnover * 1e4).mean()
    R[f"cost_{arm}"] = dict(n.mean(numeric_only=True)) | {"breakeven_bp": be,
                       "vals": [n.turnover.mean(), n.sh_0bp.mean(), n.sh_1bp.mean(), n.sh_2bp.mean(), n.sh_5bp.mean(), be]}
    rows.append(f"{lab} & {n.turnover.mean():.2f} & {n.sh_0bp.mean():.2f} & {n.sh_1bp.mean():.2f} & "
                f"{n.sh_2bp.mean():.2f} & {n.sh_5bp.mean():.2f} & {be:.2f} \\\\")
d_ = [a - b for a, b in zip(R[f"cost_{MAIN}"]["vals"], R[f"cost_{BASE}"]["vals"])]
rows += ["\\midrule", f"Gain & {d_[0]:.2f} & " + " & ".join(f"\\best{{{x:+.2f}}}" for x in d_[1:]) + " \\\\"]
write("costs", rows)

# ------------------------------------------------------------- robustness
rows = []
def rob(lab, a, b, key, **kw):
    if not (has(a) and has(b)):
        return
    r = paired(a, b, **kw)
    R[f"rob_{key}"] = r
    rows.append(f"{lab} & {r['n_ckpt']} & \\best{{{sg(r['d_ic'])}}} {ci(r)} & \\best{{{pct(r['wins_ic'])}}} \\\\")
rob("{[1]} Firm-level cross-section", MAIN + "@oc1@firm", BASE + "@oc1@firm", "firm")
rob("{[2]} Syndicated copies dropped", MAIN + "@oc1@dedup", BASE + "@oc1@dedup", "dedup")
rob("{[3]} Padding left out of the loss", "mask_ps" + A, BASE + A, "mask_pad")
seeds = [s_ for s_ in (1, 2, 3) if has(f"seed{s_}_ps{A}")]
if seeds:
    rows.append("\\multicolumn{4}{l}{[4] Random seed} \\\\")
    for s_ in seeds:
        rob(f"\\quad Seed {s_}", f"seed{s_}_ps{A}", BASE + A, f"seed{s_}")
write("robust", rows)

# ------------------------------------------------------------- write
with open(f"{RESULTS}/numbers.tex", "w") as f:
    f.write("% generated by palm/report.py -- do not edit\n")
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(R, open(f"{RESULTS}/report.json", "w"), indent=1, default=float)

print(f"tables -> {TABLES}: {sorted(os.listdir(TABLES))}")
print(f"\n=== {MAIN} vs {BASE}, next open -> next close ===")
for lab in ("ChronoGPT", "ChronoGPT-Instruct", "DatedGPT", "All backbones"):
    r = R.get(f"main_{lab}")
    if r:
        print(f"{lab:20s} n={r['n_ckpt']:2d}  IC {r['ic_b']:.2f} -> {r['ic_a']:.2f}  "
              f"gain {sg(r['d_ic'])} {ci(r)}  wins {100 * r['wins_ic']:.0f}%")
s = R["staleness"]["ic"]
print(f"staleness slope of the base IC: {s['slope'] * 1e3:+.3f}e-3/year, 90% CI [{s['lo90'] * 1e3:+.3f}, {s['hi90'] * 1e3:+.3f}]")

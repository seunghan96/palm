#!/usr/bin/env python3
"""Four macro designs for the main figure, so the layout can be chosen by eye.

The paper makes two claims and they pull a figure in opposite directions. The
first is a null, that freshness does not predict performance, and a null is
shown by a flat line with a visible spread. The second is a small positive
effect that survives pairing, and that is shown by connecting each checkpoint to
itself. No single panel does both well, so we draw four whole-figure candidates
and pick one rather than arguing about it in the abstract.

Every candidate reads from the same numbers the tables do, so choosing between
them changes only the presentation."""
import glob, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import statsmodels.formula.api as smf

from paths import ROOT, DATA, RESULTS, BIG, FIGS, PANEL, METRICS


EVAL = 2021.0
VIN = ("chrono", "PIT-4B", "datedgpt")

# One suite releases its vintages as monthly snapshots of a single continuous
# run rather than as models trained separately per cutoff, so within it a later
# vintage carries more optimizer steps as well as more text. Pooling it with the
# rest would mix two different things, so it is held out here and analysed on
# its own.
DROP = ("PIT-4B",)

# one colour per suite, kept identical across every candidate so that a reader
# comparing the four is comparing layouts and not palettes
COL = {"ChronoGPT": "#2c5f9e", "ChronoGPT-Instruct": "#2c7a4b",
       "DatedGPT": "#c0504d", "PIT-4B": "#e8a33d"}
GREY = "#8c8c8c"

PAN = pd.read_parquet(PANEL,
                      columns=["aid", "date", "fwd_ret"])
PAN["t"] = pd.to_datetime(PAN.date).dt.year + pd.to_datetime(PAN.date).dt.month / 12.0


def cutoff(m):
    g = re.findall(r"(\d{6,8})", m)
    if g:
        d = g[-1]
        return int(d[:4]) + (int(d[4:6]) if len(d) >= 6 else 12) / 12.0
    g = re.findall(r"-(\d{4})", m)
    return int(g[-1]) + 1.0 if g else None


def suite(m):
    if "chrono-gpt-instruct" in m: return "ChronoGPT-Instruct"
    if "chrono-gpt" in m: return "ChronoGPT"
    if "PIT-4B" in m: return "PIT-4B"
    if "datedgpt" in m: return "DatedGPT"
    return None


def daily(f):
    d = pd.read_parquet(f)
    if d.score.std() < 1e-9:
        return None, None
    m = d.annotator.iloc[0]
    d = d[["aid", "score"]].merge(PAN, on="aid")
    d = d[d.t >= EVAL]
    v = d.groupby("date").apply(
        lambda g: g.score.corr(g.fwd_ret, method="spearman")
        if len(g) >= 8 and g.score.nunique() >= 4 else np.nan).dropna()
    return m, (v if len(v) >= 200 else None)


def collect(sub, admissible=True):
    """Scored series per checkpoint.

    A checkpoint whose cutoff falls at or after the start of the evaluation
    window has read part of the period it is being scored on. The released
    suites carry thirteen vintages in that position, and including any of them
    would reintroduce exactly the lookahead the point-in-time suites exist to
    remove, so they are dropped by default. The boundary is closed rather than
    open because a vintage cut at the end of December 2020 has read the first
    month of a window that opens in that December."""
    out = {}
    for f in glob.glob(f"{BIG}/{sub}/*.parquet"):
        m, v = daily(f)
        if v is None or not any(x in m for x in VIN) or not suite(m):
            continue
        if suite(m) in DROP:
            continue
        c = cutoff(m)
        if admissible and (c is None or c >= EVAL):
            continue
        out[m] = v
    return out


def paired(base, sub):
    """(model, base IC, updated IC) on the days both systems scored."""
    rows = []
    for m, av in collect(sub).items():
        if m not in base:
            continue
        bv = base[m]
        j = bv.index.intersection(av.index)
        if len(j) >= 200:
            rows.append((m, bv[j].mean(), av[j].mean()))
    return rows


def ci(x):
    """Mean and its 95 percent interval, bootstrapped so no normality is assumed."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (x.mean(), x.mean(), x.mean()) if len(x) else (np.nan,) * 3
    rng = np.random.default_rng(0)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(4000)])
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def check(fig, name):
    """Refuse to ship a panel whose text collides with other text.

    Matplotlib never checks this, and the collisions that matter appear only
    after the data changes, which is exactly when nobody looks."""
    fig.canvas.draw()
    boxes = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t.get_text().strip():
                boxes.append((t.get_text()[:22], t.get_window_extent()))
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if lo <= loc <= hi and lab.get_text().strip():
                    boxes.append((lab.get_text()[:22], lab.get_window_extent()))
    bad = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b) and min(a.width, b.width) > 1:
                print(f"    OVERLAP in {name}: {boxes[i][0]!r} x {boxes[j][0]!r}")
                bad += 1
    return bad


def save(fig, name):
    n = check(fig, name)
    p = f"{FIGS}/{name}.pdf"
    fig.savefig(p, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {name}.pdf" + (f"  ({n} overlaps)" if n else ""))


# ---------------------------------------------------------------- candidate A
def _boxrect(o):
    """Display rectangle of a legend or of a text with a bbox behind it."""
    return (o.get_bbox_patch().get_window_extent()
            if hasattr(o, "get_bbox_patch") else o.get_window_extent())


def _clash(fig, boxes, pts, segs):
    """Does any boxed annotation sit on top of a drawn datum?

    Enlarging a legend is only safe up to the point where it starts hiding the
    thing the panel is about, and matplotlib will not tell us where that is, so
    we ask directly in display coordinates."""
    fig.canvas.draw()
    for r in [_boxrect(o) for o in boxes]:
        for ax, xy, pad in pts:
            for x, y in ax.transData.transform(xy):
                if r.x0 - pad <= x <= r.x1 + pad and r.y0 - pad <= y <= r.y1 + pad:
                    return True
        for ax, seg, pad in segs:
            for x0, x1, y in seg:
                (px0, py), (px1, _) = ax.transData.transform([(x0, y), (x1, y)])
                if r.y0 - pad <= py <= r.y1 + pad and \
                   px0 - pad <= r.x1 and px1 + pad >= r.x0:
                    return True
    return False


def _headroom(fig, ax, top, bot, lo, hi, pad=6.0):
    """Open a band above and below the data until the boxes clear it.

    Moving a legend around a scatter that fills the panel only trades one
    collision for another. Giving the axis room instead is the fix that keeps
    working as the legend grows, and it costs nothing but white space. The
    boxes are anchored in axes coordinates, so each change of the limits moves
    them again, which is why this iterates."""
    for _ in range(6):
        fig.canvas.draw()
        y0, y1 = ax.get_ylim()
        d = ax.transData
        need_top = need_bot = 0.0
        for o in top:
            r = _boxrect(o)
            edge = d.inverted().transform((0, r.y0 - pad))[1]
            need_top = max(need_top, hi - edge)
        for o in bot:
            r = _boxrect(o)
            edge = d.inverted().transform((0, r.y1 + pad))[1]
            need_bot = max(need_bot, edge - lo)
        if need_top <= 0 and need_bot <= 0:
            return
        ax.set_ylim(y0 - max(need_bot, 0), y1 + max(need_top, 0))


def _cand_a(B, pr, k=1.0):
    """The figure at legend scale k, with the pieces the overlap test needs."""
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.5))

    a = ax[0]
    # suites differ in average IC and in the span of vintages they cover, so a
    # pooled line through the raw points measures that composition rather than
    # staleness. Demeaning within suite is the picture of the fixed-effects
    # regression the table reports, and the two now agree by construction.
    D = B.copy()
    D["ic"] = D.ic - D.groupby("s").ic.transform("mean")
    for s, g in D.groupby("s"):
        a.scatter(g.stale, g.ic, s=42, color=COL[s], label=s, zorder=3,
                  edgecolor="white", linewidth=0.7)
    # the line has to be the fit the table reports, which carries suite fixed
    # effects. Demeaning ic alone and running OLS is a different regression, so
    # we fit with the effects and read the result off one reference suite, whose
    # own mean is what the demeaned axis has already removed.
    x = np.linspace(D.stale.min() - 0.4, D.stale.max() + 0.4, 50)
    fe = smf.ols("ic ~ stale + C(s)", data=B).fit()
    ref = B.s.value_counts().index[0]
    off = B[B.s == ref].ic.mean()
    p = fe.get_prediction(
        pd.DataFrame({"stale": x, "s": ref})).summary_frame(alpha=0.05)
    a.plot(x, p["mean"] - off, color="black", lw=1.8, zorder=4)
    a.fill_between(x, p["mean_ci_lower"] - off, p["mean_ci_upper"] - off,
                   color=GREY, alpha=0.22, zorder=1)
    a.axhline(0, color=GREY, lw=0.7, ls=":")
    a.set_xlabel("Staleness at the Decision Date (Years)")
    a.set_ylabel("Information Coefficient (Suite-Demeaned)")
    a.set_title("(a) Ineffectiveness of Retraining", fontsize=10.5)
    slope = a.text(0.03, 0.05, f"Slope {fe.params['stale']:+.5f}/Year   $p$ = "
                               f"{fe.pvalues['stale']:.2f}", transform=a.transAxes,
                   fontsize=9 * k, color="black",
                   bbox=dict(fc="white", ec=GREY, lw=0.6, pad=3.2))
    # the sign of the slope is the whole reading of the panel, and a reader
    # should not have to work it out from the axis
    key = a.text(0.028, 0.965,
                 "Slope $>0$:   Older scores better\n"
                 "Slope $=0$:   Age carries no signal\n"
                 "Slope $<0$:   Newer scores better",
                 transform=a.transAxes, fontsize=7.9 * k, va="top", ha="left",
                 linespacing=1.5, color="black",
                 bbox=dict(fc="white", ec=GREY, lw=0.6, pad=3.2))
    la = a.legend(fontsize=8 * k, loc="upper right", ncol=1, frameon=True,
                  facecolor="white", edgecolor=GREY, framealpha=1,
                  borderpad=0.42, labelspacing=0.32)
    la.get_frame().set_linewidth(0.6)

    b = ax[1]
    pr = sorted(pr, key=lambda r: r[1])
    for i, (m, lo, hi) in enumerate(pr):
        c = COL[suite(m)]
        b.plot([lo, hi], [i, i], color=c, lw=1.4, alpha=0.55, zorder=2)
        b.scatter([lo], [i], s=26, facecolor="white", edgecolor=c,
                  linewidth=1.2, zorder=3)
        b.scatter([hi], [i], s=34, color=c, zorder=4)
    b.axvline(0, color=GREY, lw=0.7, ls=":")
    b.set_yticks([])
    b.set_ylabel("Checkpoints, Sorted by Baseline")
    b.set_xlabel("Information Coefficient")
    b.set_title("(b) Effectiveness of PALM", fontsize=10.5)
    w = sum(1 for _, lo, hi in pr if hi > lo)
    b.scatter([], [], s=26, facecolor="white", edgecolor="black",
              linewidth=1.2, label="w/o PALM")
    b.scatter([], [], s=34, color="black", label="w/ PALM")
    marks = b.legend(fontsize=8.5 * k, loc="lower right", frameon=True,
                     facecolor="white", edgecolor=GREY, framealpha=1,
                     borderpad=0.42, labelspacing=0.32)
    marks.get_frame().set_linewidth(0.6)
    b.add_artist(marks)
    count = b.text(0.03, 0.965, f"Improves {w} of {len(pr)} Checkpoints",
                   transform=b.transAxes, fontsize=9 * k, va="top",
                   bbox=dict(fc="white", ec=GREY, lw=0.6, pad=3.2))
    # the panel already colours each checkpoint by suite, so name the colours,
    # listing only the suites that actually appear
    shown = [s for s in COL if any(suite(m) == s for m, _, _ in pr)]
    suites = b.legend(handles=[Line2D([], [], color=COL[s], lw=2.4, label=s)
                               for s in shown],
                      fontsize=7.6 * k, loc="upper left", frameon=True,
                      facecolor="white", edgecolor=GREY, framealpha=1,
                      bbox_to_anchor=(0.03, 0.885), handlelength=1.3,
                      handletextpad=0.5, labelspacing=0.32, borderpad=0.42)
    suites.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.22)
    # (a) is a scatter that fills its panel, so the boxes need a band opened for
    # them. (b) is a cascade sorted by baseline, which leaves its top left and
    # bottom right empty already, so opening a band there would only waste
    # height that the checkpoints should be using.
    _headroom(fig, a, [la, key], [slope], D.ic.min(), D.ic.max())
    b.set_ylim(-0.9, len(pr) - 0.1)

    # the suite legend has to start where the count box starts, and the two
    # carry different internal padding, so the offset is measured rather than
    # guessed
    fig.canvas.draw()
    want = count.get_bbox_patch().get_window_extent().x0
    have = suites.get_window_extent().x0
    inv = b.transAxes.inverted()
    dx = inv.transform((want, 0))[0] - inv.transform((have, 0))[0]
    suites.set_bbox_to_anchor((0.03 + dx, 0.885), transform=b.transAxes)

    rad = lambda s: np.sqrt(s / np.pi) * fig.dpi / 72.0
    pts = [(a, np.c_[D.stale.values, D.ic.values], rad(42)),
           (b, np.array([(v, i) for i, (_, v, _) in enumerate(pr)]), rad(26)),
           (b, np.array([(v, i) for i, (_, _, v) in enumerate(pr)]), rad(34))]
    segs = [(b, [(lo, hi, i) for i, (_, lo, hi) in enumerate(pr)], 1.0)]
    return fig, [slope, la, key, count, marks, suites], pts, segs


def cand_a(B, pr):
    """Two panels: the null on the left, the paired gain on the right.

    The legends are grown by one common factor, as far as they can go before a
    box lands on a plotted checkpoint."""
    best = 1.0
    for k in np.arange(1.0, 2.001, 0.05):
        fig, boxes, pts, segs = _cand_a(B, pr, k)
        bad = _clash(fig, boxes, pts, segs)
        # only the left panel buys space for its boxes by opening a band, so
        # only there does their size come out of the data's height
        left = boxes[1].axes
        used = sum(_boxrect(o).height for o in boxes if o.axes is left)
        bad = bad or used > 0.42 * left.bbox.height
        plt.close(fig)
        if bad:
            break
        best = k
    fig = _cand_a(B, pr, best)[0]
    print(f"  legend scale {best:.2f}")
    save(fig, "cand_a")


# ---------------------------------------------------------------- candidate B
def cand_b(B, pr):
    """One panel in the plane the decision is actually made in, cost against signal."""
    fig, a = plt.subplots(figsize=(7.4, 4.0))
    base_ic = np.mean([r[1] for r in pr])
    upd_ic = np.mean([r[2] for r in pr])
    fresh_ic, flo, fhi = ci(B[B.stale <= B.stale.min() + 1].ic.values)
    stale_ic = B[B.stale >= B.stale.max() - 6].ic.mean()

    # cost in GPU-hours, order of magnitude only. A vintage of this size is a
    # full pretraining run, the adapter is minutes on one card.
    pts = [(0.02, base_ic, "Stale Vintage\n(Already Paid For)", GREY, "o"),
           (0.35, upd_ic, "Light Update\n(Rank 2 Adapter)", "#2c7a4b", "*"),
           (3000.0, fresh_ic, "Annual Pretraining Run\n(One New Vintage)",
            "#c0504d", "s")]
    for x, y, lab, c, mk in pts:
        a.scatter([x], [y], s=300 if mk == "*" else 130, color=c, marker=mk,
                  zorder=5, edgecolor="white", linewidth=1.0)
    a.errorbar([3000.0], [fresh_ic], yerr=[[fresh_ic - flo], [fhi - fresh_ic]],
               color="#c0504d", capsize=3, lw=1.2, zorder=4)
    a.annotate("Stale Vintage\n(Already Paid For)", (0.02, base_ic),
               textcoords="offset points", xytext=(-6, -40), fontsize=9)
    a.annotate("Light Update\n(Rank 2 Adapter)", (0.35, upd_ic),
               textcoords="offset points", xytext=(-14, 14), fontsize=9,
               color="#2c7a4b", fontweight="bold")
    a.annotate("Annual Pretraining Run\n(One New Vintage)", (3000.0, fresh_ic),
               textcoords="offset points", xytext=(-138, -56), fontsize=9,
               color="#c0504d", ha="left")
    a.add_patch(FancyArrowPatch((0.02, base_ic), (0.35, upd_ic),
                                arrowstyle="-|>", mutation_scale=13,
                                color="#2c7a4b", lw=1.6, zorder=3,
                                connectionstyle="arc3,rad=-0.25"))
    a.add_patch(FancyArrowPatch((0.02, base_ic), (3000.0, fresh_ic),
                                arrowstyle="-|>", mutation_scale=13,
                                color="#c0504d", lw=1.4, ls=(0, (4, 2)),
                                zorder=3, connectionstyle="arc3,rad=0.18"))
    a.set_xscale("log")
    a.set_xlim(5e-3, 4e4)
    a.set_xlabel("Cost Of The Update (GPU-Hours, Log Scale)")
    a.set_ylabel("Information Coefficient")
    a.set_title("What Each Way of Staying Current Buys", fontsize=11, pad=14)
    a.axhline(base_ic, color=GREY, lw=0.7, ls=":")
    ys = [base_ic, upd_ic, fresh_ic, flo, fhi]
    lo, hi = min(ys), max(ys)
    pad = 0.55 * (hi - lo)
    a.set_ylim(lo - pad, hi + pad)
    save(fig, "cand_b")


# ---------------------------------------------------------------- candidate C
def cand_c(B, base, ranks):
    """Three panels: the design that makes the question answerable, then both
    claims as intervals against zero."""
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.3),
                           gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    a = ax[0]
    a.axvspan(2021, 2024, color="#2c5f9e", alpha=0.10, zorder=0)
    for i, (lab, cut) in enumerate([("Vintage 2013", 2013), ("Vintage 2016", 2016),
                                    ("Vintage 2019", 2019)]):
        y = 2 - i
        a.plot([2008, cut], [y, y], color=GREY, lw=5, solid_capstyle="butt")
        a.plot([cut, 2021], [y, y], color="#2c7a4b", lw=5, alpha=0.65,
               solid_capstyle="butt")
        a.plot([2021, 2024], [y, y], color="#2c5f9e", lw=5, alpha=0.5,
               solid_capstyle="butt")
        a.text(2007.4, y, lab, ha="right", va="center", fontsize=9)
    a.text(2012.0, 3.02, "Pretraining", fontsize=8.5, color="#4d4d4d", ha="center")
    a.text(2019.0, 3.02, "Update", fontsize=8.5, color="#2c7a4b", ha="center")
    a.text(2023.2, 3.02, "Scored", fontsize=8.5, color="#2c5f9e", ha="center")
    a.set_xlim(2002, 2025.2)
    a.set_ylim(-0.7, 3.5)
    a.set_yticks([])
    a.set_xticks([2008, 2016, 2021])
    a.set_xlabel("Calendar Year")
    a.set_title("(a) One Window, Every Vintage", fontsize=10.5)
    for sp in ("left", "right", "top"):
        a.spines[sp].set_visible(False)

    b = ax[1]
    lab, val = [], []
    for s, g in B.groupby("s"):
        if len(g) < 3:
            continue
        r = smf.ols("ic ~ stale", data=g).fit()
        lo, hi = r.conf_int().loc["stale"]
        lab.append(s); val.append((r.params["stale"], lo, hi, COL[s]))
    r = smf.ols("ic ~ stale + C(s)", data=B).fit()
    lo, hi = r.conf_int().loc["stale"]
    lab.append("Pooled"); val.append((r.params["stale"], lo, hi, "black"))
    for i, (m, lo, hi, c) in enumerate(val):
        y = len(val) - 1 - i
        b.plot([lo, hi], [y, y], color=c, lw=1.6)
        b.scatter([m], [y], s=48, color=c, zorder=3)
    b.axvline(0, color="#c0504d", lw=1.0, ls="--")
    b.set_yticks(range(len(val))[::-1])
    b.set_yticklabels(lab, fontsize=9)
    b.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    b.set_xlabel("Slope Of IC On Staleness (Per Year)")
    b.set_title("(b) Freshness Buys Nothing", fontsize=10.5)

    c = ax[2]
    for i, (rk, vals) in enumerate(ranks):
        m, lo, hi = ci(vals)
        y = len(ranks) - 1 - i
        col = "#2c7a4b" if lo > 0 else GREY
        c.plot([lo, hi], [y, y], color=col, lw=1.6)
        c.scatter([m], [y], s=48, color=col, zorder=3)
    c.axvline(0, color="#c0504d", lw=1.0, ls="--")
    c.set_yticks(range(len(ranks))[::-1])
    c.set_yticklabels([f"Rank {r}" for r, _ in ranks], fontsize=9)
    c.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    c.set_xlabel("Gain In IC From The Update")
    c.set_title("(c) A Light Update Does", fontsize=10.5)
    fig.subplots_adjust(wspace=0.36)
    save(fig, "cand_c")


# ---------------------------------------------------------------- candidate D
def cand_d(B, pr):
    """A slopegraph, which is the plot the paired test actually performs."""
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.6),
                           gridspec_kw={"width_ratios": [1.0, 1.5]})

    a = ax[0]
    for s, g in B.groupby("s"):
        gg = g.sort_values("stale")
        a.plot(gg.stale, gg.ic, color=COL[s], lw=1.3, marker="o", ms=4.5,
               label=s, alpha=0.9)
    a.axhline(B.ic.mean(), color=GREY, lw=0.8, ls=":")
    a.set_xlabel("Staleness (Years)")
    a.set_ylabel("Information Coefficient")
    a.set_title("(a) Every Vintage, One Window", fontsize=10.5)
    a.legend(fontsize=7.6, frameon=False, loc="best")

    b = ax[1]
    groups = {}
    for m, lo, hi in pr:
        groups.setdefault(suite(m), []).append((lo, hi))
    xs, ticks = [], []
    x = 0
    for s in sorted(groups):
        for lo, hi in groups[s]:
            b.plot([x, x + 0.62], [lo, hi], color=COL[s], lw=1.2, alpha=0.75,
                   marker="o", ms=3.4)
            x += 1.0
        ticks.append((x - len(groups[s]) / 2.0, s))
        xs.append(x - 0.2)
        x += 0.9
    for xv in xs[:-1]:
        b.axvline(xv, color="#dddddd", lw=0.8)
    b.set_xticks([t[0] for t in ticks])
    b.set_xticklabels([t[1] for t in ticks], fontsize=8.6)
    b.set_ylabel("Information Coefficient")
    b.set_title("(b) Without PALM (Left of Each Pair) versus With PALM (Right)",
                fontsize=10.5)
    b.tick_params(axis="x", length=0)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "cand_d")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    base = collect("base")
    print(f"baselines {len(base)}")
    B = pd.DataFrame([{"m": m, "s": suite(m), "stale": EVAL - cutoff(m),
                       "ic": v.mean()} for m, v in base.items()
                      if cutoff(m) is not None]).dropna()
    print(f"vintages with a cutoff {len(B)}")

    pr = paired(base, "palm_r2")
    if len(pr) < 4:
        pr = paired(base, "palm_r8")
        print("  rank 2 still thin, drawing with rank 8")
    print(f"paired checkpoints {len(pr)}")

    ranks = []
    for rk, sub in [(2, "palm_r2"), (4, "palm_r4"), (8, "palm_r8"),
                    (16, "palm_r16")]:
        r = paired(base, sub)
        if len(r) >= 3:
            ranks.append((rk, [h - l for _, l, h in r]))
    print(f"ranks available {[r for r, _ in ranks]}")

    cand_a(B, pr)
    cand_b(B, pr)
    if ranks:
        cand_c(B, base, ranks)
    cand_d(B, pr)

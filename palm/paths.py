#!/usr/bin/env python3
"""Where everything lives, resolved once.

Every script in this package reads and writes under a single root. That root is
the repository itself, so a fresh clone runs without editing any file. Set
PALM_ROOT to move the data and the results somewhere else, which is what you
want when the repository sits on a small disk and the scored panels do not.

    data/                the article-day panel and the alternative targets
    results/big/<arm>/   one parquet per checkpoint, the scores that arm produced
    results/cache/       one daily series per (arm, checkpoint), what every
                         number in the paper is computed from
    results/tables/      the LaTeX table bodies
    figs/                the figures
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("PALM_ROOT", REPO)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
BIG = os.path.join(RESULTS, "big")
CACHE = os.path.join(RESULTS, "cache")
TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(ROOT, "figs")

# The 252,000 article-days the paper scores, without their text. This file
# ships with the repository; restore_panel.py attaches the text from FNSPID.
PANEL_IDS = os.path.join(REPO, "data", "panel_ids.parquet")

# The panel every arm scores. One row per article-day, carrying the text, the
# ticker it names, and the return of the following session.
PANEL = os.path.join(DATA, "panel.parquet")

# The alternative targets per article-day: next open to next close (the
# paper's primary target), the overnight gap, and the session after next.
SHIFT = os.path.join(DATA, "fwd_ret_shift.parquet")

for _d in (DATA, RESULTS, BIG, CACHE, TABLES, FIGS):
    os.makedirs(_d, exist_ok=True)

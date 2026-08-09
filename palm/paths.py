#!/usr/bin/env python3
"""Where everything lives, resolved once.

Every script in this package reads and writes under a single root. That root is
the repository itself, so a fresh clone runs without editing any file. Set
PALM_ROOT to move the data and the results somewhere else, which is what you
want when the repository sits on a small disk and the scored panels do not.

    data/                one parquet, the article-day panel
    results/big/<arm>/   one parquet per checkpoint, the scores that arm produced
    results/*.csv        the per-checkpoint metrics the tables read
    figs/                the figures
"""
import os

ROOT = os.environ.get(
    "PALM_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
BIG = os.path.join(RESULTS, "big")
FIGS = os.path.join(ROOT, "figs")

# The panel every arm scores. One row per article-day, carrying the text, the
# ticker it names, and the return of the following session.
PANEL = os.path.join(DATA, "panel.parquet")

# The per-checkpoint metrics every table and figure reads.
METRICS = os.path.join(RESULTS, "metrics.csv")

for _d in (DATA, RESULTS, BIG, FIGS):
    os.makedirs(_d, exist_ok=True)

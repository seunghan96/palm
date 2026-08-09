#!/usr/bin/env python3
"""
Does date-filtered retrieval substitute for vintage pretraining?

Point-in-time suites buy their guarantee expensively: one full pretraining run
per calendar year. Retrieval offers the same guarantee cheaply -- keep one
modern model and let it read only documents published before the decision date.
The catch is that a modern model still carries whatever it memorised during its
own pretraining, and our panel ends in 2023, well inside every recent model's
window. So the substitution is only sound if that parametric residue is small
relative to what retrieval buys.

Four systems, one panel, one scoring head, one portfolio construction:

    vintage      a PIT checkpoint, parametric only          (safe by construction)
    modern       a recent model, parametric only            (measures the residue)
    modern+rag   a recent model, retrieval restricted to     (the proposal)
                 documents dated strictly before the article
    modern+leak  a recent model, retrieval unrestricted      (upper bound, unsound)

`modern+leak` is included precisely because it is not admissible: it bounds how
much retrieval can contribute when the date rule is dropped, so the gap between
it and `modern+rag` prices the constraint.

Reported per system: the information coefficient, and -- on the subset of
articles whose outcome the model could plausibly have memorised -- the paired
leakage estimate this project uses elsewhere.
"""
import argparse, sys, glob, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_loader
import score_news                      # the validated cloze/rating heads
from paths import RESULTS as OUT, PANEL


def load_index(d=f"{OUT}/embed"):
    vs, ids = [], []
    for f in sorted(glob.glob(f"{d}/shard*of*.npy")):
        if f.endswith("_aid.npy"):
            continue
        vs.append(np.load(f))
        ids.append(np.load(f.replace(".npy", "_aid.npy")))
    if not vs:
        return None, None
    V = np.concatenate(vs, 0).astype(np.float32)
    A = np.concatenate(ids, 0)
    o = np.argsort(A)
    return V[o], A[o]


def icl_context(i, panel, k, embargo, same_firm=True):
    """Labelled demonstrations drawn from articles whose outcome is already known.

    Everything tried so far gave the model more text. This gives it supervision.
    At decision date t the realised move of any article dated before t is public,
    so pairing those headlines with their outcomes is admissible under the same
    rule that admits the checkpoint itself. A weak checkpoint has never been
    trained on this task, and in-context labels are the cheapest way to teach it
    without touching a weight."""
    syms, dates = panel["symbol"].values, panel["t"].values
    txt, ret = panel["text"].values, panel["fwd_ret"].values
    cut = dates[i] - embargo * 86400.0
    ok = np.flatnonzero((dates < cut) & (syms == syms[i])) if same_firm \
        else np.flatnonzero(dates < cut)
    if len(ok) < k:
        ok = np.flatnonzero(dates < cut)
    if not len(ok):
        return ""
    pick = ok[np.argsort(dates[ok])[-k:]][::-1]
    lines = []
    for h in pick:
        if not np.isfinite(ret[h]):
            continue
        lines.append(f"News about {syms[h]}: {txt[h][:160]}\n"
                     f"Following this news, the stock price of {syms[h]} will "
                     f"{'rise' if ret[h] > 0 else 'fall'}.")
    return ("\n\n".join(lines) + "\n\n") if lines else ""


def retrieve_history(sym, syms, dates, t, k, embargo=0.0):
    """The k most recent admissible articles about the same firm.

    Similarity search over headline text returns headlines whose sentiment
    resembles the target, which is the one thing the model can already read off
    the target itself. What a stale checkpoint is missing is not sentiment but
    the firm, since it may never have encountered the company at all. Recency
    within ticker supplies that and nothing else."""
    ok = np.flatnonzero((syms == sym) & (dates < t - embargo * 86400.0))
    if not len(ok):
        return []
    return ok[np.argsort(dates[ok])[-k:]][::-1]


def retrieve_random(dates, t, k, rng):
    """k admissible documents chosen at random.

    Without this control the retrieval result is ambiguous, since prepending any
    extra financial text lengthens the prompt and shifts the readout. Random
    admissible context holds the length and the register fixed and removes only
    the relevance, so the difference isolates what similarity search contributes."""
    ok = np.flatnonzero(dates < t)
    if not len(ok):
        return []
    return rng.choice(ok, size=min(k, len(ok)), replace=False)


def retrieve(qv, V, dates, t, k, allow_future, embargo=0.0):
    """Top-k by cosine, masking anything not yet published.

    `embargo` widens the exclusion window backwards, in days. The date rule alone
    admits an article published earlier the same morning, which can already
    reflect the move we are trying to predict -- a leak that respects the letter
    of the constraint. Setting an embargo of one day removes that path, at the
    cost of discarding the most relevant context there is."""
    sim = V @ qv
    if not allow_future:
        sim = np.where(dates < t - embargo * 86400.0, sim, -np.inf)
    if not np.isfinite(sim).any():
        return []
    k = min(k, int(np.isfinite(sim).sum()))
    return np.argpartition(-sim, k - 1)[:k]


def context_for(i, panel, V, A, pos, mode, k, ctx_chars=900, embargo=0.0,
                rng=None):
    """Retrieved neighbours, rendered as a prefix to the validated prompt."""
    if mode == "none" or V is None:
        return ""
    j = pos.get(panel["aid"].values[i])
    if j is None:
        return ""
    if mode in ("icl", "iclany"):
        return icl_context(i, panel, k, embargo, same_firm=(mode == "icl"))[:ctx_chars]
    if mode == "rand":
        hits = retrieve_random(panel["t"].values, panel["t"].values[i], k, rng)
    elif mode == "wrongfirm":
        syms = panel["symbol"].values
        others = [x for x in np.unique(syms) if x != syms[i]]
        alt = others[int(rng.integers(len(others)))] if len(others) else syms[i]
        hits = retrieve_history(alt, syms, panel["t"].values,
                                panel["t"].values[i], k, embargo)
    elif mode == "mix":
        a = retrieve_history(panel["symbol"].values[i], panel["symbol"].values,
                             panel["t"].values, panel["t"].values[i], k // 2, embargo)
        b = retrieve(V[j], V, panel["t"].values, panel["t"].values[i],
                     k - k // 2, allow_future=False, embargo=embargo)
        hits = list(a) + [x for x in b if x not in set(a)]
    elif mode == "hist":
        hits = retrieve_history(panel["symbol"].values[i], panel["symbol"].values,
                                panel["t"].values, panel["t"].values[i], k, embargo)
    else:
        hits = retrieve(V[j], V, panel["t"].values, panel["t"].values[i],
                        k, allow_future=(mode == "leak"), embargo=embargo)
    hits = [h for h in hits if h != j][:k]
    if not len(hits):
        return ""
    txt = panel["text"].values
    sym = panel["symbol"].values[i]
    ctx = (f"Background on {sym} from earlier coverage:\n" +
           "\n".join("- " + txt[h][:180] for h in hits) +
           "\n\nToday's news follows.\n")
    return ctx[:ctx_chars]


@torch.no_grad()
def score(model_id, panel, V, A, mode, k, bs, head="cloze", embargo=0.0,
          ctx_chars=900):
    """Identical scoring to score_news.py, with retrieved context prepended.

    Sharing the head matters more than it looks: a bespoke prompt would make the
    retrieval systems incomparable with the vintage numbers everything else in
    this project is measured against."""
    tok, fwd, raw = model_loader.load(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pos = {a: i for i, a in enumerate(A)} if A is not None else {}
    targets = score_news.target_token_ids(tok, head)

    rng = np.random.default_rng(0)
    prompts = []
    for i in range(len(panel)):
        base = score_news.build_prompt(head, panel["symbol"].values[i],
                                       panel["text"].values[i])
        prompts.append(context_for(i, panel, V, A, pos, mode, k,
                                   ctx_chars=ctx_chars,
                                   embargo=embargo, rng=rng) + base)

    order = np.argsort([len(p) for p in prompts])          # pack similar lengths
    sc = np.zeros(len(prompts), dtype=np.float32)
    for i in range(0, len(order), bs):
        sel = order[i: i + bs]
        vals, _ = score_news.score_chunk(fwd, tok, [prompts[j] for j in sel],
                                         targets, head, "cuda")
        sc[sel] = vals
        if i % (bs * 200) == 0:
            print(f"  {i}/{len(order)}", flush=True)
    del raw, fwd
    torch.cuda.empty_cache()
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--index", default=f"{OUT}/embed")
    ap.add_argument("--ctx-chars", type=int, default=900)
    ap.add_argument("--mode", choices=["none", "rag", "leak", "rand", "hist", "mix", "wrongfirm", "icl", "iclany"], default="none")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--head", default="cloze", choices=["cloze", "rating"])
    ap.add_argument("--embargo", type=float, default=0.0,
                    help="days of extra lag required of retrieved documents")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"skip {a.out}"); sys.exit(0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    p = pd.read_parquet(a.panel,
                        columns=["aid", "date", "text", "symbol", "fwd_ret"])
    p["t"] = (pd.to_datetime(p["date"]).astype("int64") // 10**9).astype(float)
    p = p.sort_values("aid").head(a.limit).reset_index(drop=True)
    if a.shards > 1:
        # keep the retrieval index over the whole panel and only split
        # the rows we score, so a shard sees the same candidate set
        keep = (np.arange(len(p)) % a.shards) == a.shard
        p = p[keep].reset_index(drop=True)

    V, A = (None, None)
    if a.mode != "none":
        V, A = load_index(a.index)
        if V is None:
            raise SystemExit("no embeddings yet -- run embed_panel.py first")
        keep = np.isin(A, p["aid"].values)
        V, A = V[keep], A[keep]
        p = p[p["aid"].isin(A)].reset_index(drop=True)

    s = score(a.model, p, V, A, a.mode, a.k, a.batch_size, a.head, a.embargo,
              a.ctx_chars)
    pd.DataFrame({"aid": p["aid"], "date": p["date"], "symbol": p["symbol"],
                  "score": s, "annotator": a.model, "mode": a.mode,
                  "k": a.k, "head": a.head, "embargo": a.embargo}).to_parquet(a.out, index=False)
    print(f"WROTE {a.out}  n={len(s)}")

#!/usr/bin/env python3
"""Adapt a stale checkpoint on admissible text, instead of pretraining a new one.

Retrieval and in-context labels both failed, and in hindsight both were asking a
language model to use information it was shown rather than information it has.
The thing a vintage is missing is not a document, it is the more mundane drift in
how financial text is written: which tickers exist, which phrases carry weight,
which companies are being discussed at all.

That is what continued pretraining fixes, and continued pretraining is exactly
what the point-in-time suites pay for once a year at full cost. The question this
script asks is whether the cheap version suffices: take the old checkpoint, fit a
low-rank adapter on the news published between its cutoff and the decision date,
and leave every original weight alone.

The adaptation is admissible by construction. Only text dated before the
evaluation window enters it, and no return label is used at any point, so the
result is still a model that has read nothing from its own future.

The comparison that matters is against a modern model with no such guarantee. If
an adapter closes that gap, the annual pretraining run is not the only way to
stay current, and it is roughly three orders of magnitude more expensive than the
alternative."""
import argparse, math, os, sys
import numpy as np
import pandas as pd
import torch

from paths import ROOT, DATA, RESULTS, BIG, FIGS, PANEL, METRICS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_loader


class LoRALinear(torch.nn.Module):
    """Rank-r update on a frozen linear layer.

    Written by hand rather than pulled from peft because two of the three suites
    ship custom model classes that peft cannot introspect, and the wrapper only
    needs to work for nn.Linear."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        # match the frozen layer's device and dtype. The base model is already on
        # the GPU by the time adapters are injected, and fresh Parameters default
        # to CPU float32.
        w = base.weight
        self.a = torch.nn.Parameter(torch.zeros(r, d_in, device=w.device,
                                                dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(d_out, r, device=w.device,
                                                dtype=torch.float32))
        torch.nn.init.normal_(self.a, std=1e-2)
        self.scale = alpha / r

    def forward(self, x):
        d = torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), self.a), self.b) * self.scale
        return self.base(x) + d.to(x.dtype)


def _block(name):
    """The block index a module sits in, read off its dotted path."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def inject(model, r, alpha, keys=("c_attn", "c_proj", "q_proj", "v_proj",
                                  "query", "value", "c_fc", "up_proj"),
           depth="all"):
    # freeze everything first. Injecting adapters does not freeze the layers we
    # did not touch, and leaving them trainable would be a full finetune wearing
    # a LoRA label.
    for q in model.parameters():
        q.requires_grad = False
    idx = [b for b in (_block(nm) for nm, _ in model.named_modules()) if b is not None]
    top = max(idx) if idx else 0
    lo, hi = (0, top // 3) if depth == "first" else \
             (top - top // 3, top) if depth == "last" else (0, top)
    n = 0
    for name, mod in list(model.named_modules()):
        b = _block(name)
        if depth != "all" and (b is None or not (lo <= b <= hi)):
            continue
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and any(k in child_name for k in keys):
                setattr(mod, child_name, LoRALinear(child, r, alpha))
                n += 1
    return n


def train(model, fwd, tok, texts, steps, bs, lr, maxlen, device="cuda",
          seed=0):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        idx = rng.integers(0, len(texts), bs)
        batch = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                    truncation=True, max_length=maxlen)
        ids = batch["input_ids"].to(device)
        if ids.shape[1] < 8:
            continue
        logits = fwd(ids)          # the loader hides three calling conventions
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}/{steps} loss {np.mean(losses[-50:]):.4f}", flush=True)
    model.eval()
    return float(np.mean(losses[-50:])) if losses else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--cutoff", type=float, required=True,
                    help="the checkpoint's own cutoff, in fractional years")
    ap.add_argument("--eval-from", type=float, default=2021.0,
                    help="adaptation text stops here; scoring starts here")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--head", default="cloze")
    ap.add_argument("--shuffle-text", action="store_true",
                    help="scramble word order in the adaptation corpus")
    ap.add_argument("--seed", type=int, default=0,
                    help="controls adapter initialization and batch order, so "
                         "that run to run variance can be separated from the "
                         "effect being measured")
    ap.add_argument("--incremental", action="store_true",
                    help="adapt one calendar year at a time, carrying the same "
                         "adapter forward, instead of one pass over the whole "
                         "admissible window")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="keep only the most recent fraction of the admissible "
                         "window, to ask how much text the update needs")
    ap.add_argument("--full", action="store_true",
                    help="train every parameter instead of injecting an adapter. "
                         "This is continued pretraining on the eligible text, "
                         "which is the alternative PALM is proposed to replace, "
                         "run on the same text and the same budget")
    ap.add_argument("--depth", default="all", choices=["all", "first", "last"],
                    help="restrict the adapter to the first or the last third of "
                         "the blocks, to locate the update in depth")
    ap.add_argument("--keys", default="",
                    help="comma separated projection names to adapt. Empty "
                         "means the default attention and feed-forward set")
    ap.add_argument("--fit-to", type=float, default=None,
                    help="upper bound of the adaptation corpus. Defaults to the "
                         "decision date, which is the point-in-time setting. "
                         "Setting it past the decision date breaks eligibility "
                         "on purpose, and measures what an adapter allowed to "
                         "read the evaluation window would reach")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"skip {a.out}"); sys.exit(0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    p = pd.read_parquet(a.panel, columns=["aid", "date", "text", "symbol", "fwd_ret"])
    p["t"] = pd.to_datetime(p.date).dt.year + pd.to_datetime(p.date).dt.month / 12.0
    top = a.eval_from if a.fit_to is None else a.fit_to
    fit = p[(p.t > a.cutoff) & (p.t < top)]
    if a.frac < 1.0:
        # keep the most recent slice, which is the text a practitioner would
        # reach for first if they could only afford to read some of it
        lo = top - a.frac * (top - a.cutoff)
        fit = fit[fit.t >= lo]
    ev = p[p.t >= a.eval_from].reset_index(drop=True)
    print(f"adaptation texts {len(fit):,} ({a.cutoff:.1f} to {top:.1f})   "
          f"evaluation articles {len(ev):,}", flush=True)
    if len(fit) < 2000 or len(ev) < 5000:
        # Leave a marker. Without it the scheduler sees a missing output, emits
        # the job again, and a combination that can never produce data is
        # retried for as long as the queue runs.
        open(a.out + ".nodata", "w").write(
            f"{len(fit)} adaptation texts, {len(ev)} evaluation articles\n")
        print("not enough data on one side"); sys.exit(0)

    torch.manual_seed(a.seed)
    tok, fwd, model = model_loader.load(a.model, trainable=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.full:
        # No adapter at all. Every weight is trainable, which is the comparison
        # a reader asks for, since it is the same text and the same budget spent
        # on the whole model rather than on a low-rank update beside it.
        for q in model.parameters():
            q.requires_grad = True
        n = -1
        print("full finetune, no adapter injected", flush=True)
    else:
        kw = dict(depth=a.depth)
        if a.keys:
            kw["keys"] = tuple(k for k in a.keys.split(",") if k)
        n = inject(model, a.rank, 2 * a.rank, **kw)
        if n == 0:
            print("no layer matched the requested keys"); sys.exit(0)
    tp = sum(q.numel() for q in model.parameters() if q.requires_grad)
    tot = sum(q.numel() for q in model.parameters())
    print(f"LoRA on {n} layers, {tp/1e6:.2f}M trainable of {tot/1e6:.0f}M "
          f"({100*tp/tot:.3f}%)", flush=True)

    texts = fit.text.tolist()
    rng = np.random.default_rng(a.seed)
    if a.shuffle_text:
        texts = [" ".join(rng.permutation(t.split())) for t in texts]
    if a.incremental:
        # A desk does not wait until the decision date to start reading. It
        # updates once a year, on the year just finished, and carries the same
        # adapter forward. The total step budget is held equal to the one shot
        # setting so the comparison is about scheduling, not about compute.
        years = list(range(int(math.ceil(a.cutoff)), int(a.eval_from)))
        years = years or [int(a.cutoff)]
        per = max(50, a.steps // len(years))
        print(f"incremental over {len(years)} years, {per} steps each",
              flush=True)
        loss = float("nan")
        for y in years:
            sel = fit[(fit.t >= y) & (fit.t < y + 1)]
            if len(sel) < 200:
                continue
            chunk = sel.text.tolist()
            if a.shuffle_text:
                chunk = [" ".join(rng.permutation(t.split())) for t in chunk]
            loss = train(model, fwd, tok, chunk, per, a.batch_size, a.lr,
                         a.maxlen, seed=a.seed)
            print(f"  year {y} loss {loss:.4f}", flush=True)
    else:
        loss = train(model, fwd, tok, texts, a.steps, a.batch_size, a.lr,
                     a.maxlen, seed=a.seed)
    print(f"final loss {loss:.4f}", flush=True)

    import score_news
    targets = score_news.target_token_ids(tok, a.head)
    prompts = [score_news.build_prompt(a.head, s, t)
               for s, t in zip(ev.symbol, ev.text)]
    order = np.argsort([len(x) for x in prompts])
    sc = np.zeros(len(prompts), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), a.score_batch):
            sel = order[i: i + a.score_batch]
            v, _ = score_news.score_chunk(fwd, tok, [prompts[j] for j in sel],
                                          targets, a.head, "cuda")
            sc[sel] = v
            if i % (a.score_batch * 400) == 0:
                print(f"  scoring {i}/{len(order)}", flush=True)
    pd.DataFrame({"aid": ev.aid, "date": ev.date, "symbol": ev.symbol,
                  "score": sc, "annotator": a.model, "mode": "adapt",
                  "rank": a.rank, "steps": a.steps, "frac": a.frac, "seed": a.seed,
                  "incremental": a.incremental,
                  "eval_from": a.eval_from}).to_parquet(a.out, index=False)
    print(f"WROTE {a.out}  n={len(sc)}")

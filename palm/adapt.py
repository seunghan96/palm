#!/usr/bin/env python3
"""Adapt a stale checkpoint on admissible text, instead of pretraining a new one.

Take the old checkpoint, fit a low-rank adapter on the news published between
its cutoff and the decision date, and leave every original weight alone. The
adaptation is admissible by construction: only text dated before the evaluation
window enters it, and no return label is used at any point, so the result is
still a model that has read nothing from its own future.

The same script scores the result, so one run produces one parquet of scores
per checkpoint and arm. The flags that depart from the point-in-time setting
exist to measure a bound or a control, and the paper reports them as such:

  --fit-window LO HI   train on LO < t <= HI instead of (cutoff, decision date).
                       Setting HI to the cutoff gives the same-domain control:
                       FNSPID text the vintage's era already covers, at the same
                       budget, so any gain it produces is domain adaptation
                       rather than news of the missing period.
  --fit-to T           let the adapter read past the decision date. Breaks
                       eligibility on purpose; the paper reports it as a bound.
  --eval-to T          score only eval_from <= t < T. The validation runs read a
                       window that ends where the test window begins.
  --sft                replace the language-modeling objective with a supervised
                       one on the same readout, fitted to the sign of returns
                       realized before the decision date.
  --full               train every weight instead of injecting an adapter, on
                       the same text and the same budget.
  --no-train           score the checkpoint as released through this same path,
                       so a paired comparison holds the batching fixed.
  --save-adapter DIR   keep the adapter's weights after training.
  --load-adapter DIR   skip training and score with a saved adapter instead.
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PANEL
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


def adapter_state(model):
    """The adapter's own tensors, keyed by the layer they sit beside."""
    return {f"{name}.{k}": getattr(mod, k).detach().cpu()
            for name, mod in model.named_modules() if isinstance(mod, LoRALinear)
            for k in ("a", "b")}


def save_adapter(model, path, meta):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    save_file(adapter_state(model), os.path.join(path, "adapter.safetensors"))
    json.dump(meta, open(os.path.join(path, "adapter.json"), "w"), indent=1)


def load_adapter(model, path):
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(path, "adapter.json")))
    n = inject(model, meta["rank"], 2 * meta["rank"], keys=tuple(meta["keys"]),
               depth=meta["depth"])
    st = load_file(os.path.join(path, "adapter.safetensors"))
    mods = {name: mod for name, mod in model.named_modules() if isinstance(mod, LoRALinear)}
    assert set(st) == {f"{nm}.{k}" for nm in mods for k in ("a", "b")}, \
        "the saved adapter does not match this checkpoint's layers"
    with torch.no_grad():
        for nm, mod in mods.items():
            mod.a.copy_(st[f"{nm}.a"].to(mod.a.device))
            mod.b.copy_(st[f"{nm}.b"].to(mod.b.device))
    model.eval()
    return n, meta


def train(model, fwd, tok, texts, steps, bs, lr, maxlen, device="cuda",
          seed=0, mask_pad=False):
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
        tgt = ids[:, 1:]
        if mask_pad:
            # The paper's objective (mask_pad=False) also scores the eos tokens
            # used as right padding, so part of its loss is predicting padding.
            # This variant keeps every real token and the one eos that ends a
            # text, and ignores the rest of the padding.
            am = batch["attention_mask"].to(device)
            keep = am[:, :-1].bool()           # predict from a real position only
            tgt = tgt.masked_fill(~keep, -100)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}/{steps} loss {np.mean(losses[-50:]):.4f}", flush=True)
    model.eval()
    return float(np.mean(losses[-50:])) if losses else float("nan")


def train_sft(model, fwd, tok, prompts, labels, targets, steps, bs, lr,
              device="cuda", seed=0, max_len=768):
    """Fit the cloze log-odds to the sign of the realized return.

    The readout is the one scoring uses, so the only thing that differs from
    PALM is the objective. Returns are those of articles inside the fit window,
    all realized before the decision date."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    up_ids, dn_ids = targets
    losses = []
    for step in range(steps):
        idx = rng.integers(0, len(prompts), bs)
        encs = [tok(prompts[i], add_special_tokens=False, truncation=True,
                    max_length=max_len)["input_ids"] for i in idx]
        L = max(len(e) for e in encs)
        inp = torch.full((len(encs), L), tok.eos_token_id, dtype=torch.long)
        for k, e in enumerate(encs):
            inp[k, :len(e)] = torch.tensor(e)          # right padding, as in scoring
        inp = inp.to(device)
        logits = fwd(inp).float()
        last = logits[torch.arange(len(encs), device=device),
                      torch.tensor([len(e) - 1 for e in encs], device=device)]
        lp = torch.nn.functional.log_softmax(last, dim=-1)
        z = torch.logsumexp(lp[:, up_ids], 1) - torch.logsumexp(lp[:, dn_ids], 1)
        y = torch.tensor(labels[idx], device=device, dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  sft step {step}/{steps} loss {np.mean(losses[-50:]):.4f}", flush=True)
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
    ap.add_argument("--fit-window", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="train on LO < t <= HI. Overrides --cutoff/--fit-to "
                         "for the fit set only")
    ap.add_argument("--eval-to", type=float, default=None,
                    help="score only eval_from <= t < eval_to")
    ap.add_argument("--sft", action="store_true",
                    help="supervised objective on the cloze readout, labels "
                         "from returns realized inside the fit window")
    ap.add_argument("--ctx", default="none", choices=["none", "hist"],
                    help="prepend same-firm history at scoring time")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mask-pad", action="store_true",
                    help="leave padding out of the language-modeling loss")
    ap.add_argument("--no-train", action="store_true",
                    help="score the unadapted checkpoint through this same path, "
                         "so a paired comparison holds the scored articles and "
                         "their batching fixed")
    ap.add_argument("--save-adapter", default=None, metavar="DIR",
                    help="write the trained adapter (safetensors + json) here")
    ap.add_argument("--load-adapter", default=None, metavar="DIR",
                    help="score with this saved adapter instead of training one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"skip {a.out}"); sys.exit(0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    p = pd.read_parquet(a.panel, columns=["aid", "date", "text", "symbol", "fwd_ret"])
    p["t"] = pd.to_datetime(p.date).dt.year + pd.to_datetime(p.date).dt.month / 12.0
    top = a.eval_from if a.fit_to is None else a.fit_to
    if a.fit_window is not None:
        lo, hi = a.fit_window
        # the fit set may never reach the evaluation window, whatever is asked
        assert hi < a.eval_from, "fit window reads the evaluation window"
        fit = p[(p.t > lo) & (p.t <= hi)]
        top = hi
    else:
        fit = p[(p.t > a.cutoff) & (p.t < top)]
    if a.sft:
        # a label is only admissible if the return it records was realized
        # before the decision date. fwd_ret is the next session's return, so
        # the last month before the window is dropped to keep a margin.
        fit = fit[fit.t < a.eval_from - 1.0 / 12 + 1e-9]
    if a.frac < 1.0:
        # keep the most recent slice, which is the text a practitioner would
        # reach for first if they could only afford to read some of it
        lo = top - a.frac * (top - a.cutoff)
        fit = fit[fit.t >= lo]
    ev = p[p.t >= a.eval_from]
    if a.eval_to is not None:
        ev = ev[ev.t < a.eval_to]
    ev = ev.reset_index(drop=True)
    lo_print = a.fit_window[0] if a.fit_window is not None else a.cutoff
    print(f"adaptation texts {len(fit):,} ({lo_print:.3f} to {top:.3f})   "
          f"evaluation articles {len(ev):,}", flush=True)
    skip_train = a.no_train or a.load_adapter is not None
    if (len(fit) < 2000 and not skip_train) or len(ev) < 5000:
        # Leave a marker. Without it a scheduler sees a missing output, emits
        # the job again, and a combination that can never produce data is
        # retried for as long as the queue runs.
        open(a.out + ".nodata", "w").write(
            f"{len(fit)} adaptation texts, {len(ev)} evaluation articles\n")
        print("not enough data on one side"); sys.exit(0)

    torch.manual_seed(a.seed)
    tok, fwd, model = model_loader.load(a.model, trainable=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    keys = tuple(k for k in a.keys.split(",") if k) or None
    meta = None
    if a.load_adapter:
        n, meta = load_adapter(model, a.load_adapter)
        print(f"loaded adapter from {a.load_adapter} ({n} layers)", flush=True)
    elif a.no_train:
        n = 0
        print("no training, scoring the checkpoint as released", flush=True)
    elif a.full:
        # No adapter at all. Every weight is trainable, which is the comparison
        # a reader asks for, since it is the same text and the same budget spent
        # on the whole model rather than on a low-rank update beside it.
        for q in model.parameters():
            q.requires_grad = True
        n = -1
        print("full finetune, no adapter injected", flush=True)
    else:
        kw = dict(depth=a.depth)
        if keys:
            kw["keys"] = keys
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
    if skip_train:
        loss = float("nan")
    elif a.incremental:
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
    elif a.sft:
        import score_news
        tg = score_news.target_token_ids(tok, "cloze")
        sp = [score_news.build_prompt("cloze", s, t)
              for s, t in zip(fit.symbol, fit.text)]
        lab = (fit.fwd_ret.values > 0).astype(np.float32)
        loss = train_sft(model, fwd, tok, sp, lab, tg, a.steps, a.batch_size,
                         a.lr, seed=a.seed)
    else:
        loss = train(model, fwd, tok, texts, a.steps, a.batch_size, a.lr,
                     a.maxlen, seed=a.seed, mask_pad=a.mask_pad)
    print(f"final loss {loss:.4f}", flush=True)

    if a.save_adapter and not (a.no_train or a.full or a.load_adapter):
        default_keys = inject.__defaults__[0]
        save_adapter(model, a.save_adapter, dict(
            model=a.model, cutoff=a.cutoff, eval_from=a.eval_from, rank=a.rank,
            steps=a.steps, lr=a.lr, seed=a.seed, maxlen=a.maxlen,
            batch_size=a.batch_size, depth=a.depth, keys=list(keys or default_keys),
            fit_lo=float(fit.t.min()), fit_hi=float(fit.t.max()), n_fit=len(fit),
            final_loss=loss, sft=a.sft, mask_pad=a.mask_pad,
            shuffle_text=a.shuffle_text, incremental=a.incremental, frac=a.frac))
        print(f"saved adapter to {a.save_adapter}", flush=True)

    import score_news
    targets = score_news.target_token_ids(tok, a.head)
    prompts = [score_news.build_prompt(a.head, s, t)
               for s, t in zip(ev.symbol, ev.text)]
    if a.ctx == "hist":
        # Same-firm history retrieval: the k most recent articles on the same
        # ticker, 180 characters each, 900 in all, so that the composed arm
        # reads exactly the context the retrieval baseline reads. Only articles
        # dated strictly before the scored article's own day are admissible.
        secs = (pd.to_datetime(p.date).astype("int64") // 10**9).astype(float).values
        syms, txt = p.symbol.values, p.text.values
        by = {}
        for i, sy in enumerate(syms):
            by.setdefault(sy, []).append(i)
        by = {k: np.array(v)[np.argsort(secs[v], kind="stable")] for k, v in by.items()}
        evs = (pd.to_datetime(ev.date).astype("int64") // 10**9).astype(float).values
        for n, (sy, tt) in enumerate(zip(ev.symbol.values, evs)):
            idx = by.get(sy, np.array([], dtype=int))
            ok = idx[secs[idx] < tt]
            hits = ok[-a.k:][::-1]
            if len(hits):
                ctx = (f"Background on {sy} from earlier coverage:\n" +
                       "\n".join("- " + txt[h][:180] for h in hits) +
                       "\n\nToday's news follows.\n")[:900]
                prompts[n] = ctx + prompts[n]
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
    rank, steps, lr = (meta["rank"], meta["steps"], meta["lr"]) if meta else (a.rank, a.steps, a.lr)
    pd.DataFrame({"aid": ev.aid, "date": ev.date, "symbol": ev.symbol,
                  "score": sc, "annotator": a.model, "mode": "adapt",
                  "rank": rank, "steps": steps, "frac": a.frac, "seed": a.seed,
                  "incremental": a.incremental,
                  "eval_from": a.eval_from, "lr": lr, "sft": a.sft, "mask_pad": a.mask_pad, "ctx": a.ctx,
                  "fit_lo": float(fit.t.min()) if len(fit) else float("nan"),
                  "fit_hi": float(fit.t.max()) if len(fit) else float("nan"),
                  "n_fit": len(fit), "final_loss": loss}).to_parquet(
                      a.out + ".tmp", index=False)
    os.replace(a.out + ".tmp", a.out)   # a killed worker never leaves a half file
    print(f"WROTE {a.out}  n={len(sc)}")

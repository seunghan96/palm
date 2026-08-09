#!/usr/bin/env python3
"""
Vintage-annotator scoring: label FNSPID news with a knowledge-cutoff-controlled LLM.

Design constraints discovered in the checkpoint audit
(papers/PIT_checkpoint_technical_audit.md):
  * PIT-4B has NO KV cache -> free generation is O(n^2). Never generate.
    Instead take ONE forward pass and read first-token probabilities.
  * forward() ignores attention_mask -> RIGHT padding only, never left.
  * output_hidden_states unsupported -> hook the last block if embeddings needed.

Two scoring heads:
  --head rating   : FNSPID-compatible 1..5 sentiment.
                    s = sum_k k*P(tok"k") / sum_k P(tok"k")   (probability-weighted)
                    Works with instruct vintages (PIT-4B-FT, chrono-gpt-instruct).
  --head cloze    : base-LM friendly. Prompt ends "... the stock price will"
                    s = logP(" rise") - logP(" fall")  (log-odds of direction)
                    Works with base vintages, no instruction-following needed.

Both are single-forward, so cost is 1 pass/article regardless of head.
"""
import os
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_loader

# Verbatim reconstruction of the FNSPID prompt (paper Figure 2), few-shot
# demonstrations included. Dropping them collapses the output distribution onto
# "1" for weakly instruction-following models -- verified on ChronoGPT-Instruct,
# which put 77% of the digit mass on "1" without them.
RATING_TMPL = (
    "You are a financial expert with stock recommendation experience. "
    "Based on a specific stock, score for range from 1 to 5, where 1 is negative, "
    "2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive.\n"
    "News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%\n"
    "Score: 5\n"
    "News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%\n"
    "Score: 1\n"
    "News to Stock Symbol -- MSFT: Microsoft (MSFT) price has no change\n"
    "Score: 3\n"
    "News to Stock Symbol -- {sym}: {text}\n"
    "Score:"
)

CLOZE_TMPL = (
    "News about {sym}: {text}\n"
    "Following this news, the stock price of {sym} will"
)


def build_prompt(head, sym, text, max_chars=1500):
    text = " ".join(str(text).split())[:max_chars]
    return (RATING_TMPL if head == "rating" else CLOZE_TMPL).format(sym=sym, text=text)


def target_token_ids(tok, head):
    if head == "rating":
        # first token of " 1".." 5" and "1".."5" — take the bare-digit variants
        ids = []
        for k in "12345":
            a = tok.encode(k, add_special_tokens=False)
            b = tok.encode(" " + k, add_special_tokens=False)
            ids.append((a[0], b[0]))
        return ids
    else:
        # The reported set, plus two disjoint alternatives. The gain should not
        # depend on which words are used to name the direction, and the only way
        # to find out is to name it differently.
        SETS = {
            "base": ([" rise", " increase", " go up", " gain"],
                     [" fall", " decrease", " go down", " drop"]),
            "alt":  ([" climb", " advance", " strengthen", " rally"],
                     [" slide", " retreat", " weaken", " tumble"]),
            "plain": ([" up", " higher", " better", " stronger"],
                      [" down", " lower", " worse", " weaker"]),
        }
        up, dn = SETS.get(os.environ.get("CLOZE_WORDS", "base"), SETS["base"])

        def first_distinguishing(words):
            """The first token that actually differs between candidates.

            SentencePiece tokenizers emit the leading space as its own token, so
            taking element [0] returns that same space id for every candidate and
            the up minus down contrast collapses to exactly zero. GPT-2 BPE folds
            the space into the word and does not have this problem, which is why
            it went unnoticed until a Llama-tokenized suite was added."""
            seq = [tok.encode(w, add_special_tokens=False) for w in words]
            i = 0
            while (i + 1 < min(len(x) for x in seq)
                   and len({x[i] for x in seq}) == 1):
                i += 1
            return [x[i] for x in seq]

        both = first_distinguishing(up + dn)
        return both[:len(up)], both[len(up):]


@torch.no_grad()
def score_chunk(fwd, tok, prompts, targets, head, device, max_len=768):
    encs = [tok(p, add_special_tokens=False, truncation=True,
                max_length=max_len)["input_ids"] for p in prompts]
    L = max(len(e) for e in encs)
    pad = tok.eos_token_id
    inp = torch.full((len(encs), L), pad, dtype=torch.long)
    for i, e in enumerate(encs):
        inp[i, : len(e)] = torch.tensor(e)          # RIGHT padding (safe: causal)
    inp = inp.to(device)

    logits = fwd(inp).float()
    # last REAL token of each sequence predicts the answer token
    idx = torch.tensor([len(e) - 1 for e in encs], device=device)
    last = logits[torch.arange(len(encs), device=device), idx]   # (B, V)
    lp = F.log_softmax(last, dim=-1)

    if head == "rating":
        # sum the bare-digit and space-digit variants in probability space
        probs = []
        for a, b in targets:
            p = lp[:, a].exp() + lp[:, b].exp()
            probs.append(p)
        P = torch.stack(probs, dim=1)                # (B,5)
        mass = P.sum(1)
        Pn = P / mass.clamp_min(1e-12).unsqueeze(1)
        k = torch.arange(1, 6, device=device, dtype=P.dtype)
        score = (Pn * k).sum(1)
        extra = {"prob_mass": mass.cpu().numpy(),
                 **{f"p{i+1}": Pn[:, i].cpu().numpy() for i in range(5)}}
    else:
        up_ids, dn_ids = targets
        up = torch.logsumexp(lp[:, up_ids], dim=1)
        dn = torch.logsumexp(lp[:, dn_ids], dim=1)
        score = up - dn                              # log-odds
        extra = {"logp_up": up.cpu().numpy(), "logp_dn": dn.cpu().numpy()}
    return score.cpu().numpy(), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", required=True, help="parquet with date, symbol, text")
    ap.add_argument("--head", choices=["rating", "cloze"], default="cloze")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(a.panel)
    if a.limit:
        df = df.iloc[: a.limit]
    print(f"panel rows={len(df):,}", flush=True)

    tok, fwd, raw = model_loader.load(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    targets = target_token_ids(tok, a.head)

    prompts = [build_prompt(a.head, s, t) for s, t in zip(df["symbol"], df["text"])]
    # sort by length so batches are homogeneous -> less padding waste
    order = np.argsort([len(p) for p in prompts])
    scores = np.zeros(len(prompts), dtype=np.float32)
    extras = {}
    t0 = time.time()
    for i in range(0, len(order), a.batch_size):
        sel = order[i : i + a.batch_size]
        sc, ex = score_chunk(fwd, tok, [prompts[j] for j in sel], targets,
                             a.head, "cuda", a.max_len)
        scores[sel] = sc
        for k, v in ex.items():
            extras.setdefault(k, np.zeros(len(prompts), dtype=np.float32))[sel] = v
        if (i // a.batch_size) % 100 == 0:
            done = i + len(sel)
            r = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(order)}  {r:.1f}/s  "
                  f"eta={(len(order)-done)/max(r,1e-9)/60:.1f}min", flush=True)

    idcols = [c for c in ["aid", "date", "symbol"] if c in df.columns]
    out = df[idcols].copy()
    out["score"] = scores
    for k, v in extras.items():
        out[k] = v
    out["annotator"] = a.model
    out["head"] = a.head
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"WROTE {a.out}  rows={len(out):,}  "
          f"score mean={scores.mean():.4f} sd={scores.std():.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

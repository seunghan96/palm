# PALM: Point-in-Time Adaptation for Financial Language Models

**Paper:** [arXiv:2609.30316](https://arxiv.org/abs/2609.30316) ·
**Venue:** ICAIF '26 Workshop on LLMs and Generative AI for Finance (AI4F),
Milan, November 2026 — **oral presentation**

This repository is the code behind the workshop paper, which is the final
version of the work. Everything below, the panel, the settings and the
analysis, matches arXiv v2. The code behind the earlier main-track submission
(August 2026, one shared rank-2 setting, a different panel and target) is kept
at the tag [`v1-icaif-submission`](https://github.com/seunghan96/palm/tree/v1-icaif-submission) for the
record and is superseded.

---

## The claims

**Point-in-time (PIT) language models** are pretrained only on text published
before a documented **cutoff**, and one such checkpoint, or **vintage**, is
released per calendar year. A backtest run at a given date uses a vintage whose
cutoff precedes it, and that vintage has therefore never seen the period it is
tested on. The guarantee is real, and *every additional year of coverage costs
another pretraining run*.

- **A newer checkpoint is not detectably better on the decision it informs.**
  Once every vintage is scored on one fixed evaluation window, so that the
  checkpoint is the only thing that varies, the information coefficient shows
  no decay with age that we can detect: the slope on staleness is
  +0.088 × 10⁻³ per year, 90% interval [−0.140, +0.321], across 44 vintages
  of three PIT families whose cutoffs span two decades.
- **A small adapter does the work of the run.** PALM fits a low-rank adapter
  on financial text the checkpoint is allowed to read, leaves *every
  pretrained weight frozen*, and raises the pooled IC from 1.63 to 3.40 × 10⁻³
  (+1.77, 95% interval [−0.15, +3.71]) on the next session's open-to-close
  return, improving **41 of 44 checkpoints**.
- **The gain comes from the financial domain rather than from recency.** On
  the vintages where both can be run, an adapter fed text from *before* the
  cutoff gains as much (+1.80) as one fed the text the vintage is missing
  (+1.37). The point-in-time constraint therefore comes without a detectable
  cost, and the adapter costs a small fraction of a pretraining run.

Because the adapter reads nothing from the evaluation window, *an adapted
checkpoint remains eligible wherever the original one was*.

---

## What the pipeline produces

| Path | Contents |
| --- | --- |
| `data/panel_ids.parquet` | **ships with the repository**: the 252,000 article-days the paper scores, without their text |
| `data/panel.parquet` | the same rows with the text attached from FNSPID, one row per article-day |
| `data/fwd_ret_shift.parquet` | the alternative targets per article-day; `fwd_ret_oc1` is the paper's primary one |
| `results/big/<arm>/*.parquet` | one file per checkpoint, the scores that arm produced |
| `results/cache/<arm>@oc1/*.parquet` | one daily series (IC, return, turnover) per checkpoint and arm |
| `results/tables/*.tex`, `results/numbers.tex` | the table bodies and the macros the paper's prose uses |
| `results/report.json` | every estimate and interval, for checking |
| `figs/ws_main.pdf` | the main figure |

An **arm** is one experimental condition. `base_test` is the checkpoint as
released, the `Base` row of every table. `sel_ps` is PALM at the reported
setting. The rest, listed in `configs/arms.txt`, are the domain control, the
two bounds, the alternatives that also train, retrieval, and robustness.

---

## Install

```bash
pip install -r requirements.txt
```

- Scoring and adaptation need a **CUDA GPU** (14 GB for the 1.5B models,
  24 GB for DatedGPT, 36 GB for full finetuning).
- Everything downstream of `results/big/` runs on a laptop.

By default the repository is also the data root. Point `PALM_ROOT` somewhere
else when the scored panels do not fit next to the code:

```bash
export PALM_ROOT=/scratch/palm
```

---

## Data

The panel is built from [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
(Dong et al., 2024), released under **CC BY-NC 4.0**. Its text is not
redistributed here. Download two files from that release into `data/`:

| File | What it is |
| --- | --- |
| `Stock_news/nasdaq_exteral_data.csv` | the news with UTC timestamps (23 GB) |
| `Stock_price/full_history.zip` | the daily price history |

The paper scores **252,000 article-days, 25,200 per calendar year from 2014
to 2023**, over 1,099 tickers. FNSPID carries no article identifier, so
`data/panel_ids.parquet` names each row by its timestamp, its ticker and the
SHA-1 of the text the pipeline composed for it (the headline, followed by
FNSPID's own summary where one exists), together with the trading day it is
credited to, the publisher and the next-session return. `restore_panel.py`
attaches the text and refuses to write anything if a single row is missing,
so the restored panel has the same rows, in the same order, under the same
`aid` as the paper.

`palm/build_panel.py` is the step that produced the wider pool the paper's
rows were drawn from (top 1,200 tickers by coverage, at most 60 articles per
ticker and year, the 20:00 UTC rule below). It is kept so the construction
can be read and rerun; the shipped list is what makes the paper's exact
sample reproducible.

---

## Reproduce

```bash
bash run/verify.sh             # a one-minute check that the tree is wired
bash run/00_data.sh            # restore the panel and derive the targets
bash run/01_score_base.sh      # score the 44 checkpoints as released
bash run/02_adapt.sh           # fit PALM at the reported setting and score again
bash run/03_metrics.sh         # the daily series behind every number
bash run/04_tables_figures.sh  # the tables, the macros and the main figure
```

Notes on each stage:

- **Step 0** reads the 23 GB news file once (about fifteen minutes) and the
  price history, and also writes `results/audit.json`, the point-in-time audit
  of the panel.
- **Steps 1 and 2** download checkpoints from the Hugging Face Hub on first
  use and *skip any output already on disk*, so they are safe to interrupt
  and resume.
- **Steps 1 and 2 are the expensive ones.** One checkpoint takes about
  **20 minutes** to score on one L40S at 1.5B parameters, and adaptation adds
  about the same again. Budget about **30 GPU-hours** for the two arms the
  headline table needs.
- **Steps 3 and 4** read no model and finish in minutes. `NBOOT` (default
  2,000) sets the bootstrap draws behind every interval.

### The reported setting

The adapter setting was chosen **once per backbone, on a validation window
only** (decision date end of November 2017, scored through November 2020,
which ends where the test window begins), by a rule fixed before any
validation result existed. `configs/settings.txt` holds the result:

| Backbone | Rank | Steps | Learning rate |
| --- | --- | --- | --- |
| ChronoGPT | 4 | 1,200 | 10⁻⁴ |
| ChronoGPT-Instruct | 1 | 600 | 10⁻⁴ |
| DatedGPT | 16 | 300 | 10⁻⁴ |

To redo the selection, run the validation grid (14 vintages, 21 settings
each, about 60 GPU-hours) and the rule:

```bash
bash run/02_adapt.sh val        # val_base and val_r{1,2,4,8,16}_s{300,600,1200}_lr1e-4, plus two rates
python3 palm/select_config.py   # -> results/selection.csv, results/selection_by_suite.csv
```

### The other arms

Every arm of the paper is one line of `configs/arms.txt`, and one command:

```bash
ARM=dom_ps     bash run/02_adapt.sh   # same budget, text from before the cutoff
ARM=shuf_ps    bash run/02_adapt.sh   # bound: the words scrambled
ARM=oracle_ps  bash run/02_adapt.sh   # bound: text read past the decision date
ARM=full_ft    bash run/02_adapt.sh   # every weight trained, same text and budget
ARM=sft_r2     bash run/02_adapt.sh   # supervised on returns realized before the date
ARM=base_hist  bash run/02_adapt.sh   # same-firm retrieval at scoring time
ARM=seed1_ps   bash run/02_adapt.sh   # another draw of the adapter
```

Rerun steps 3 and 4 afterwards; tables whose arms are absent are left out
rather than failing.

### Keeping the adapters

The pipeline scores each adapter in the process that trained it, and keeps
only the scores. To keep the adapter as well:

```bash
SAVE_ADAPTERS=1 bash run/02_adapt.sh          # -> results/adapters/sel_ps/<model>/
python3 palm/adapt.py --model <id> --cutoff <c> \
    --load-adapter results/adapters/sel_ps/<model> --out <file>   # score with it later
```

An adapter is a `safetensors` file of the low-rank factors beside each adapted
projection (15 MB at rank 4 on a 1.5B model), plus a `json` of the setting it
was trained with. Training is seeded and scoring is a single forward pass, so
a saved adapter reproduces the scores of the run that produced it.

---

## Layout

```
palm/
  paths.py            where data, results and figures live
  restore_panel.py    FNSPID + data/panel_ids.parquet -> the paper's panel
  build_panel.py      FNSPID -> the wider article-day pool (for reference)
  shift_returns.py    the alternative targets: next open -> next close, and others
  model_loader.py     loads each suite's checkpoints behind one interface
  score_news.py       the cloze readout: one forward pass per article
  adapt.py            fit the adapter on eligible text, then score; also
                      the base, domain, bound, full-finetuning, supervised
                      and retrieval arms, and saving or loading an adapter
  daily.py            scored panel -> one row per date: IC, return, turnover
  stats.py            month-block x checkpoint bootstrap; paired gains, slopes
  select_config.py    the validation rule that picked the setting
  audit.py            the point-in-time audit of the panel and of every arm
  report.py           the tables, the macros and the headline numbers
  main_figure.py      the main figure
configs/
  checkpoints.txt     every checkpoint, its cutoff, eligibility, validation pool
  settings.txt        the per-backbone setting the paper reports
  arms.txt            every arm and the flags that define it
data/
  panel_ids.parquet   the 252,000 article-days the paper scores
run/                  the six steps above
```

---

## How the evaluation avoids look-ahead

Three rules, **enforced in code rather than by convention**.

- **An article belongs to the day it could have been read.** An article is
  credited to a trading day only when its timestamp falls *before that day's
  close*, taken conservatively at 20:00 UTC. 99.6% of FNSPID timestamps carry
  a date and no time, so the paper's primary target is the **next session's
  open-to-close return**, which is tradable at the next open whenever the
  article actually appeared on its stamped date. `shift_returns.py` also
  derives the overnight gap and the session after next, and the alignment
  table reports all four.
- **A checkpoint is eligible only if its cutoff precedes the decision date.**
  `configs/checkpoints.txt` marks which vintages qualify. The ones cut at the
  end of 2020 or later have read into the evaluation window and are *scored
  but never reported*.
- **The adapter reads only what the checkpoint was already allowed to read.**
  `adapt.py` takes `--cutoff` and `--eval-from` and trains on the interval
  between them. *No return label enters the objective* except in the
  supervised baseline, where every label is realized before the decision
  date. The `--fit-to` flag that would break this exists only to measure the
  bound that breaking it buys, and `audit.py` checks every scored arm on disk
  for an adapter that read past its date.

---

## Notes

- Scoring is a **single forward pass with no sampling**, so a checkpoint
  returns identical scores on identical input, and *every difference between
  two runs is attributable to the checkpoint rather than to decoding*.
- The suites differ in what they support. **PIT-4B has no KV cache and ignores
  the attention mask**, so `score_news.py` never generates and pads on the
  right only. `model_loader.py` hides these differences behind one interface.
- `results/big/` reaches a few hundred gigabytes if the whole arm grid is run.
  Only `results/cache/` is needed to regenerate the tables and figures.

---

## Citation

```bibtex
@inproceedings{lee2026palm,
  title     = {{PALM}: Point-in-Time Adaptation for Financial Language Models},
  author    = {Lee, Seunghan and Seo, Jun and Lee, Jaehoon and Kang, Junhyeok and
               Han, Sangjun and Yoo, Sungdong and Kim, Minjae and Lim, Tae Yoon and
               Kang, Dongwan and Choi, Hwanil and Lee, Soonyoung and Ahn, Wonbin},
  booktitle = {ICAIF '26 Workshop on LLMs and Generative AI for Finance (AI4F)},
  year      = {2026},
  note      = {arXiv:2609.30316}
}
```

# PALM: Point-in-Time Adaptation for Financial Language Models

**Point-in-time (PIT) language models** are pretrained only on text published
before a documented **cutoff**, and one such checkpoint, or **vintage**, is
released per calendar year. A backtest run at a given date uses a vintage whose
cutoff precedes it, and that vintage has therefore never seen the period it is
tested on. The guarantee is real, and *every additional year of coverage costs
a full pretraining run*.

This repository holds the code behind two results.

- **The newer checkpoint is not the better one.** Once every vintage is scored
  on one fixed evaluation window, so that the checkpoint is the only thing that
  varies, *staleness does not degrade downstream performance in any suite*.
- **A small adapter does the work of the run.** PALM fits a **rank-2 LoRA
  adapter** on the text published between a checkpoint's cutoff and the
  decision date, leaves *every pretrained weight frozen*, and raises the
  information coefficient on **40 of 44 checkpoints** across three suites.

Because the adapter reads nothing from the evaluation window, *an adapted
checkpoint remains eligible wherever the original one was*.

---

## What the pipeline produces

| Path | Contents |
| --- | --- |
| `data/panel.parquet` | one row per article-day, with the next session's return |
| `results/big/<arm>/*.parquet` | one file per checkpoint, the scores that arm produced |
| `results/metrics.csv` | one row per (checkpoint, arm): IC, return, Sharpe |
| `figs/*.pdf` | the figures in the paper |

An **arm** is one experimental condition.

- `base` is the unadapted checkpoint, the `-` row of every table.
- `palm_r2` is PALM at the reported setting.
- The rest vary *one choice at a time*: capacity, training budget, decision
  date, how much of the corpus is read, which projections carry the update.

`configs/arms.txt` lists them all.

---

## Install

```bash
pip install -r requirements.txt
```

- Scoring and adaptation need a **CUDA GPU**.
- Everything downstream of `results/metrics.csv` runs on a laptop.

By default the repository is also the data root. Point `PALM_ROOT` somewhere
else when the scored panels do not fit next to the code:

```bash
export PALM_ROOT=/scratch/palm
```

---

## Reproduce

```bash
bash run/verify.sh             # a one-minute check that the tree is wired
bash run/00_data.sh            # build the panel from FNSPID
bash run/01_score_base.sh      # score every checkpoint, no adapter
bash run/02_adapt.sh           # fit PALM and score again
bash run/03_metrics.sh         # IC, return and Sharpe per checkpoint and arm
bash run/04_tables_figures.sh  # the LaTeX rows and the figures
```

Notes on each stage:

- **Step 0** needs two files from the
  [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) release in
  `data/`: `nasdaq_exteral_data.csv` and `full_history.zip`.
- **Steps 1 and 2** download checkpoints from the Hugging Face Hub on first use
  and *skip any output already on disk*, so they are safe to interrupt and
  rerun.
- **Steps 1 and 2 are the expensive ones.** One checkpoint takes roughly
  **20 minutes** to score on a single L40S at 1.5B parameters, and adaptation
  adds about the same again. The reported pool is 44 checkpoints, so budget
  about **60 GPU-hours** for the two arms the headline table needs.
- **Steps 3 and 4** read no model and finish in under a minute.

### Running one sensitivity arm

`run/02_adapt.sh` reads its setting from the environment, so an arm is one line:

```bash
RANK=8            ARM=palm_r8   bash run/02_adapt.sh   # capacity
STEPS=200         ARM=steps200  bash run/02_adapt.sh   # training budget
EVAL_FROM=2019.0  ARM=d2019     bash run/02_adapt.sh   # a different decision date
```

Add the arm to `configs/arms.txt` and rerun steps 3 and 4 to see it in the
tables. *Arms listed there but absent from disk are skipped*, so the file can
carry the full grid while only part of it has been run.

---

## Layout

```
palm/
  paths.py          where data, results and figures live
  build_panel.py    FNSPID -> the article-day panel
  model_loader.py   loads each suite's checkpoints behind one interface
  score_news.py     one forward pass per article, cloze or rating head
  adapt.py          fit the adapter on eligible text, then score
  rag_score.py      the read-only baselines: retrieval and in-context labels
  metrics.py        scored panels -> results/metrics.csv
  paper_tables.py   metrics.csv -> the LaTeX rows in the paper
  main_figure.py    staleness, and the paired before and after comparison
  rank_curve.py     the adapter capacity curve
  qual3.py          cumulative gain, and the vocabulary the update moves
configs/
  checkpoints.txt   every checkpoint, its cutoff, and whether it is eligible
  arms.txt          every arm, its directory, and its decision date
run/                the six steps above
```

---

## How the evaluation avoids look-ahead

Three rules, **enforced in code rather than by convention**.

- **An article belongs to the day it could have been read.**
  `build_panel.py` credits an article to a trading day only when its timestamp
  falls *before that day's close*, taken conservatively at 20:00 UTC. The
  target is the return of the **following** session, so a signal is never
  scored against a bar it could have moved.
- **A checkpoint is eligible only if its cutoff precedes the decision date.**
  `configs/checkpoints.txt` marks which vintages qualify. The ones cut at the
  end of 2020 or later have read into the evaluation window and are *scored but
  never reported*.
- **The adapter reads only what the checkpoint was already allowed to read.**
  `adapt.py` takes `--cutoff` and `--eval-from` and trains on the interval
  between them. *No return label enters the objective at any point.* The
  `--fit-to` flag that would break this exists only to measure the ceiling that
  breaking it buys, which the paper reports as a **bound rather than as a
  method**.

---

## Notes

- Scoring is a **single forward pass with no sampling**, so a checkpoint
  returns identical scores on identical input, and *every difference between
  two runs is attributable to the checkpoint rather than to decoding*.
- The suites differ in what they support. **PIT-4B has no KV cache and ignores
  the attention mask**, so `score_news.py` never generates and pads on the
  right only. `model_loader.py` hides these differences behind one interface.
- `results/` reaches a few hundred gigabytes if the whole arm grid is run. Only
  **`metrics.csv`** is needed to regenerate the tables and figures.

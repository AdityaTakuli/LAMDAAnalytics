# Artifact README (anonymous TMLR submission)

This archive accompanies *Overlapping-Window Targets Manufacture Forecasting
Skill*. It is meant to be posted as an anonymized OpenReview / anonymous Git
link at submission. Until that link exists, the paper does not claim a public
release.

## Contents

- `dataset/data/four_year_2021_2024/processed/nodes_monthly.csv` — fused panel
- `artifact/DATASHEET.md` — sources, licenses, span, reporter, HS basket, row counts
- `dataset/phase0_t1_ladder.py` — target family T0 / T1 / \(H_k\) / T4
- `results/canonical_numbers.csv` — frozen table source of truth
- `results/phase07/` — matched-budget graph seeds and ensembles
- `dataset/*.py` — horizon sweep, DM tests, AR(1) decomposition, canonical rebuild
- `figures/` — overlap and per-origin persistence plots
- `reproduce.sh` — single command to rebuild canonical numbers from frozen graph runs

## Single command

```bash
./reproduce.sh
```

That rebuilds tabular / AR / exogenous numbers and writes
`results/canonical_numbers.csv`. It does **not** retrain GCN/TGN (those runs
are frozen in `results/phase07/`). To retrain graphs at the paper budget
(80 epochs, patience 15, five seeds):

```bash
cd dataset
python graph_convergence.py
```

Graph training is slow on CPU.

## Environment

Python 3.10+ with pandas, numpy, scikit-learn, lightgbm. The paper used
`backend/.venv`. See `dataset/requirements.txt` and `dataset/requirements-train.txt`.

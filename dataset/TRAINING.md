# Model training guide

How to train, evaluate, and reproduce the country-month supply-chain risk
models on a Linux or Windows machine, with or without a CUDA GPU.

Everything here runs from the `dataset/` directory. Nothing in this guide
downloads data or calls an external API — training reads only the fused tables
that the acquisition and fusion stages already produced (see `README.md`).

---

## Contents

1. [What gets trained](#1-what-gets-trained)
2. [Prerequisites](#2-prerequisites)
3. [Install — Linux with CUDA](#3-install--linux-with-cuda)
4. [Install — Windows with CUDA](#4-install--windows-with-cuda)
5. [Install — CPU only](#5-install--cpu-only)
6. [Verify the install](#6-verify-the-install)
7. [Train](#7-train)
8. [Read the results](#8-read-the-results)
9. [Every option](#9-every-option)
10. [Reproducing a run exactly](#10-reproducing-a-run-exactly)
11. [Loading a checkpoint](#11-loading-a-checkpoint)
12. [Tests](#12-tests)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What gets trained

| Script | Purpose |
| --- | --- |
| `check_training_env.py` | Preflight. Checks packages, GPU, and data readiness. Trains nothing. |
| `train_models.py` | The training entry point. Trains every model, writes every artifact. |
| `training/` | The package the entry points use: data, models, engine, metrics, reporting. |
| `tests/test_training.py` | 26 tests covering targets, splits, leakage rules, metrics, and an end-to-end run. |

**Supervised unit.** One country-month row. Features come from month `T`; the
target is derived from inbound Comtrade flow at month `T+1`.

```text
baseline    = median of inbound_flow_usd over the 12 months ending at T
future      = inbound_flow_usd at T+1
contraction = (future - baseline) / baseline
label(tau)  = 1 when contraction < -tau
```

A row is supervised only when the future value is observed **and** the baseline
is positive. Rows that fail either condition are excluded from training and
from evaluation. They are never converted into negative examples.

**Two tasks.**

| `--task` | Target | Use when |
| --- | --- | --- |
| `regression` (default) | `contraction`, a signed ratio | Always available. This is the defensible target for the current data. |
| `classification` | `label(tau)`, binary | Only when the training months contain both classes. |

The default is `regression` because the binary label is extremely rare in this
dataset: 2 positives at `tau=0.30` and none at `0.35` or `0.40` in the 2024
profile, and 15 of 626 valid targets in the four-year profile. `train_models.py`
refuses a classification run whose training partition has a single class, and
tells you what to do instead. That refusal is deliberate — see
[Troubleshooting](#13-troubleshooting).

**Six models, one split.**

| Name | Kind | Notes |
| --- | --- | --- |
| `gcn` | graph | Snapshot GCN. Same-month edges, no temporal memory. |
| `tgn` | graph | Temporal Graph Network. GRU memory carried across months. |
| `tgn_no_memory` | graph | The TGN ablation. Isolates what the memory contributes. |
| `constant_prevalence` / `train_median` | baseline | Predicts one constant fitted on training months. |
| `hand_weighted_linear` | baseline | The fixed deployment heuristic. Classification only. |
| `logistic_regression` / `ridge_regression` | baseline | Linear model on the same features. |

---

## 2. Prerequisites

* **Python 3.10 – 3.13.** Check with `python --version`.
* **The fused tables must already exist** for the profile you want to train.
  Verify with `python check_training_env.py --config config.yaml`. If they are
  missing, build them first — from `dataset/`:

  ```bash
  python fuse_dataset.py --config config.yaml
  python build_graph.py  --config config.yaml
  ```

* **For CUDA:** an NVIDIA GPU with a driver new enough for the CUDA runtime you
  install. Confirm the driver works before installing anything Python:

  ```bash
  nvidia-smi
  ```

  The `CUDA Version` shown in that output is the **maximum** runtime your driver
  supports. Pick a PyTorch CUDA wheel at or below it (`cu121`, `cu124`, `cu126`).

> **You do not need a GPU.** These models are small — roughly 2.6k parameters
> for the GCN and 21k for the TGN, over 20 country nodes. A full run on the
> 2024 profile takes about 15 seconds on a laptop CPU. The GPU path exists so
> the same scripts scale to larger graphs and longer histories without change;
> on the current data a GPU is not faster, because each monthly snapshot is far
> too small to fill one.

---

## 3. Install — Linux with CUDA

```bash
# 1. From the repository root, enter the dataset directory
cd LAMDAAnalytics/dataset

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA. Match the wheel to your driver:
#      cu121 -> driver supports CUDA 12.1+
#      cu124 -> driver supports CUDA 12.4+   (a good default)
#      cu126 -> driver supports CUDA 12.6+
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Install the remaining training dependencies
pip install -r requirements-train.txt

# 5. Confirm the GPU is visible to PyTorch
python check_training_env.py --device cuda
```

Step 5 must print `CUDA available : True`, name your GPU, and report
`Matmul smoke test: passed`. If it does not, stop and fix the install before
training — see [Troubleshooting](#13-troubleshooting).

---

## 4. Install — Windows with CUDA

Use **PowerShell** (not `cmd.exe`) so the activation script below works.

```powershell
# 1. From the repository root, enter the dataset directory
cd LAMDAAnalytics\dataset

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA (same wheel choice as Linux)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Install the remaining training dependencies
pip install -r requirements-train.txt

# 5. Confirm the GPU is visible to PyTorch
python check_training_env.py --device cuda
```

If PowerShell blocks the activation script with
`running scripts is disabled on this system`, allow it for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In `cmd.exe`, activate with `.venv\Scripts\activate.bat` instead.

Every command in this guide is identical on Windows apart from the activation
line and backslashes in paths. The scripts resolve all their own paths relative
to `dataset/`, so they work from any working directory and on either path
separator.

---

## 5. Install — CPU only

Works on Linux, Windows, and macOS (including Apple Silicon).

```bash
cd LAMDAAnalytics/dataset
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
python check_training_env.py
```

---

## 6. Verify the install

Two checks. Run both once on any new machine.

**a. Environment and data readiness**

```bash
python check_training_env.py --config config.yaml
```

```text
Packages
  numpy        2.1.1                    ok
  pandas       2.2.3                    ok
  torch        2.5.1+cu124              ok
  ...
Device
  PyTorch          : 2.5.1+cu124 (CUDA build: 12.4)
  CUDA available   : True
    GPU 0          : NVIDIA RTX A4000 (sm_86, 15.99 GB)
  Selected device  : cuda:0 (CUDA is available)
  Matmul smoke test: passed
Data (config.yaml)
  nodes    found  .../data/one_year_2024/processed/nodes_monthly.csv
  edges    found  .../data/one_year_2024/processed/edges_monthly.csv
  240 country-month rows, 12 months, 20 countries, 7404 edges
  187 valid targets across 11 supervised months
  positives by tau: {'0.30': 2, '0.35': 0, '0.40': 0}
```

Exit code `0` means ready, `1` means a package or device problem, `2` means the
data is not ready.

**b. End-to-end self-test**

This trains every model on a small synthetic panel with a known, deliberately
imperfect signal, then asserts that the artifacts appear and that a graph model
beats chance. It touches no data profile and writes nothing you have to clean
up.

```bash
python train_models.py --self-test --device cuda      # or --device cpu
```

The last line must read `SELF-TEST PASSED on device cuda:0`.

---

## 7. Train

### The 2024 profile (12 months, 20 countries)

```bash
python train_models.py --config config.yaml
```

That uses everything in the `model_training:` block of `config.yaml`:
continuous regression, the split `2024-01…07 / 2024-08…09 / 2024-10…11`, 30
epochs with early stopping, and `--device auto`.

### The four-year profile (48 months)

```bash
python train_models.py --config config_4year.yaml --device cuda
```

Split: train `2021-01…2022-12`, validation `2023`, test `2024`. December 2024 is
dropped automatically because its next-month target would need January 2025;
the exclusion is recorded in `diagnostics/split.json`.

### Common variations

```bash
# See exactly what a run would do, and write nothing
python train_models.py --config config.yaml --dry-run

# Force the GPU (fails loudly rather than silently using the CPU)
python train_models.py --config config.yaml --device cuda

# Force a specific GPU on a multi-GPU box
python train_models.py --config config.yaml --device cuda:1

# Both tasks, into two separate run directories
python train_models.py --config config.yaml --task both --overwrite

# Only the TGN and its ablation, longer schedule
python train_models.py --config config.yaml --models tgn,tgn_no_memory --epochs 100 --patience 15

# An explicit split, overriding the config
python train_models.py --config config_4year.yaml \
    --train-range 2021-01:2022-06 \
    --validation-range 2022-07:2023-06 \
    --test-range 2023-07:2024-11

# Somewhere else on disk, leaving the profile untouched
python train_models.py --config config.yaml --output-dir /tmp/run-a
```

Re-running into a directory that already holds artifacts is refused. Pass
`--overwrite` to replace it, or `--output-dir` to write elsewhere. This is on
purpose: two runs must never be mixed in one results directory.

---

## 8. Read the results

Default location: `<results_dir>/model_training/<task>/`, so for the 2024
profile in regression mode:

```text
data/one_year_2024/results/model_training/regression/
├── README.md                    # what each file is, written per run
├── run_summary.json             # START HERE: options, environment, split, all metrics
├── config_used.yaml             # the configuration this run actually saw
├── training.log                 # the full console log
├── metrics/
│   ├── comparison.csv           # one row per (model, split) — the results table
│   └── <model>.json             # full metric block per model
├── predictions/
│   └── <model>.csv              # model, split, month, node_id, target, score
├── checkpoints/
│   └── <model>.pt               # weights + kwargs + features + normaliser + split
├── diagnostics/
│   ├── leakage_audit.md         # what was fitted on what, and why nothing leaks
│   ├── target_summary.json      # target validity, contraction distribution, class balance
│   ├── label_cross_check.json   # re-derived labels vs the labels stored in the table
│   ├── class_balance.json       # classification runs only
│   ├── data_validation.json     # row counts, grid completeness, feature ranges
│   ├── split.json               # the exact months per partition, and any dropped month
│   └── baseline_notes.json      # how each baseline was fitted
└── plots/
    ├── target_distribution.png
    ├── loss_curves.png
    ├── model_comparison.png
    └── prediction_distributions.png
```

### Which number is the headline

| Task | Report | Why |
| --- | --- | --- |
| regression | `rmse`, `mae`, `spearman_r` | Compare against `train_median`. A model that cannot beat a constant has learned nothing. |
| classification | `average_precision` (PR-AUC), then `roc_auc` | The positive class is rare, so PR-AUC is the informative one. Accuracy is not. |

### Reading the numbers honestly

* A metric printed as `null` with a `note` is **undefined**, not zero. ROC-AUC,
  PR-AUC, precision, recall, and F1 are undefined when a partition contains a
  single class; the run says so rather than filling in a number.
* `n` counts country-months with an **observable** target. With the 2024
  profile that is 119 training rows and 34 test rows. Differences of a few
  points at that size are noise, not evidence.
* Test metrics come from one scoring pass with the validation-selected weights.
  They are not a tuning target. Selecting a threshold, a `tau`, or an epoch by
  test performance invalidates the number.
* Always read `metrics/comparison.csv` together with
  `diagnostics/class_balance.json` (classification) or the
  `contraction_by_split` block of `diagnostics/target_summary.json`
  (regression).

---

## 9. Every option

```text
python train_models.py [options]

Inputs
  --config PATH               YAML profile (default: config.yaml)
  --task {classification,regression,both}
  --tau FLOAT                 Contraction threshold for the binary label
  --baseline-min-periods INT  Months required for the rolling baseline median

Split
  --split-mode {date_ranges,counts}
  --train-range START:END     e.g. 2021-01:2022-12
  --validation-range START:END
  --test-range START:END
  --train-months INT          counts mode
  --validation-months INT     counts mode
  --test-months INT           counts mode

Models and optimisation
  --models LIST               comma separated: gcn,tgn,tgn_no_memory
  --baselines LIST            comma separated; default is all applicable
  --epochs INT
  --learning-rate FLOAT
  --weight-decay FLOAT
  --grad-clip FLOAT           gradient-norm clip; 0 disables
  --patience INT              early-stopping patience in epochs
  --threshold FLOAT           decision threshold for classification metrics
  --regression-loss {huber,mse}
  --huber-beta FLOAT

Runtime
  --device {auto,cpu,cuda,cuda:N}
  --seed INT
  --deterministic             request deterministic kernels
  --log-level {DEBUG,INFO,WARNING,ERROR}

Outputs and gates
  --output-dir PATH
  --overwrite
  --allow-degenerate          continue despite a single-class training split
  --dry-run                   validate, print the plan, write nothing
  --json                      print the run summary as JSON

Self-test
  --self-test
  --self-test-epochs INT
```

**Precedence:** command line > the `model_training:` block in the config >
built-in defaults.

**Exit codes:** `0` completed · `1` unexpected error · `2` data or split cannot
support the run · `3` requested device unavailable.

### The `model_training:` config block

Both `config.yaml` and `config_4year.yaml` carry this block. No other script
reads it, so editing it cannot affect the acquisition or fusion stages.

```yaml
model_training:
  task: "regression"            # or "classification"
  tau: 0.35                     # binary threshold; fixed in advance, never tuned on test
  baseline_window: 12           # months in the baseline median
  baseline_min_periods: 1       # 1 for the 12-month profile, 12 for the four-year profile
  models: ["gcn", "tgn", "tgn_no_memory"]
  split:
    mode: "date_ranges"         # or "counts"
    train:      ["2024-01", "2024-07"]
    validation: ["2024-08", "2024-09"]
    test:       ["2024-10", "2024-11"]
  epochs: 30
  learning_rate: 0.005
  patience: 8
  device: "auto"
  seed: 7
```

`baseline_min_periods` matters. The 2024 profile is only twelve months long, so
a full twelve-month baseline is impossible and `1` is used, which reproduces the
labels already stored in `nodes_monthly.csv`. The four-year profile uses the
strict `12`; as a result the first eleven months of 2021 have no baseline and
are dropped from training, which the run reports in `diagnostics/split.json`.

---

## 10. Reproducing a run exactly

```bash
python train_models.py --config config.yaml --seed 7 --deterministic --device cpu
```

* `--seed` seeds Python, NumPy, and PyTorch, on CPU and on every CUDA device.
* `--deterministic` additionally requests deterministic kernels.

**Bitwise reproducibility is guaranteed on CPU only.** The GCN's mean
aggregation uses `index_add_`, which has no deterministic CUDA implementation;
on GPU the flag is applied in warn-only mode so the run still completes, and
`run_summary.json` records the reduced guarantee under `determinism`. Two GPU
runs with the same seed will agree closely but may differ in the last decimals.

Every run stores what it needs to be repeated: `config_used.yaml`, the resolved
options, the exact split, package versions, GPU model, and the driver's CUDA
build — all in `run_summary.json`.

---

## 11. Loading a checkpoint

Checkpoints are self-describing. They carry the model class, its constructor
kwargs, the feature order, the fitted normaliser, the edge scales, the node
order, and the split — so no config file is needed to reload one.

```python
import sys
sys.path.insert(0, "dataset")          # or run from inside dataset/

from training.models import load_checkpoint

model, meta = load_checkpoint(
    "data/one_year_2024/results/model_training/regression/checkpoints/tgn.pt",
    device="cuda",                      # or "cpu"
)

print(meta["task"])                     # 'regression'
print(meta["features"])                 # the 7 feature names, in model input order
print(meta["normalizer"]["mean"])       # standardisation fitted on training months
print(meta["split"]["test"])            # the months this model was tested on
```

To score new months, standardise features with `meta["normalizer"]` in
`meta["features"]` order — remembering that `inventory_days_proxy` is
sign-flipped first, as `sign_flipped` in the normaliser records — and order the
nodes as `meta["node_order"]`. The reference implementation of that
transformation is `training.data.FeatureStandardizer` and
`training.data.build_month_batches`.

---

## 12. Tests

```bash
python tests/test_training.py           # no test runner required
python -m pytest tests/test_training.py -v   # if pytest is installed
```

26 tests. They cover the contraction formula against a hand-computed case, the
rule that invalid targets never become negatives, split disjointness and
forward chaining, train-only fitting of the standardiser and edge scales, table
validation, the leakage rule that no target column appears in the feature list,
`null`-not-zero metrics for degenerate partitions, device resolution, checkpoint
round-tripping, and a full end-to-end run.

Run them after any change to `training/`, and once on any new machine.

---

## 13. Troubleshooting

**`--device cuda` fails with "the installed PyTorch is a CPU-only build"**

`pip install torch` without an index URL installs the CPU wheel. Reinstall:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The last command must print a version like `2.5.1+cu124`, a CUDA version, and
`True`.

**`nvidia-smi` works but `torch.cuda.is_available()` is `False`**

The CUDA wheel is newer than the driver. Check the `CUDA Version` in the
`nvidia-smi` header and install a wheel at or below it (`cu121` instead of
`cu126`, for example). On Windows, also confirm you are in the virtual
environment you installed into — `where python` should point inside `.venv`.

**"The training partition contains a single class"**

This is the gate working, not a bug. The binary label is genuinely too rare in
this dataset. In order of preference:

1. Use the continuous target: `--task regression`.
2. Train on a window that contains positives (the four-year profile).
3. For a plumbing check only, `--allow-degenerate`. The resulting metrics are
   `N/A` by construction and are not a benchmark.

Do **not** lower `tau` after seeing results, and do not rebalance or synthesise
labels. Both would make the reported number meaningless.

**"Output directory already contains artifacts"**

Intentional. Use `--overwrite` to replace that run, or `--output-dir` to keep
both.

**"Missing nodes table" / "Missing edges table"**

The profile has not been fused yet. From `dataset/`:

```bash
python fuse_dataset.py --config config.yaml
python build_graph.py  --config config.yaml
```

**"The configured split needs N supervised months but only M are available"**

The requested partitions are longer than the data. Lower the counts, switch to
`mode: date_ranges`, or extend the window. The message prints the months that
actually exist.

**`ModuleNotFoundError: No module named 'common'`**

You are running a file inside `training/` directly. Use the entry points —
`train_models.py` or `check_training_env.py` — which put `dataset/` on the
import path.

**Warning: "Feature(s) [...] are constant across the whole table"**

Real and worth acting on. In the 2024 profile `weather_anomaly_7d` and
`global_risk` are constant, so they contribute nothing. Their standardised value
is pinned to 0 rather than dividing by a zero scale. Check the corresponding
ingest stage if you expected variation.

**Training is slower on GPU than on CPU**

Expected at this scale. Twenty nodes per snapshot cannot fill a GPU, and the
TGN's temporal attention loops over nodes in Python, so kernel-launch overhead
dominates. Use `--device cpu` for the current profiles; the CUDA path is there
for larger graphs.

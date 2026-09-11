# LAMDA codebase PRD (read this before editing)

This file is the product and repository map for agents (Claude, Cursor, Codex).
The root `README.md` describes the **live demo app**. This file describes **the
whole repo as it actually is**, including the research paper that is now the
scientific claim.

If these two disagree, **this file and `Research Paper/paper.tex` win**.
`changes.md` is lab notes and can lag.

---

## 1. What this repository is

LAMDAAnalytics is **two products in one git tree**. They share a seven-feature
schema and a TGN idea. They do **not** share validated extractors, and they
must not be mixed in claims.

| Product | Purpose | Status vs the paper |
|---|---|---|
| **Paper / corpus** | Methodological negative result for TMLR | **The claim.** Frozen numbers. |
| **Live app** | FastAPI + LangGraph + React demo | **Not a contribution.** Extractors unvalidated. |

**Paper claim (one sentence):** overlapping-window disruption targets manufacture
most reported skill; that bias is known in financial econometrics; importing it
onto a public trade panel inverts model rankings; after the overlap is removed,
nothing beats the target's own lags, an AR(1,12) at \(R^2=0.215\) (univariate AR(1): 0.171).

**Live-app claim (one sentence):** a lane query is mapped onto seven features by
LLM/SERP/weather agents, then scored by a TGN checkpoint or a weighted fallback.
That pipeline is **explicitly excluded** from the paper.

Venue: **TMLR**. Format: official `tmlr.sty` in `Research Paper/`. Not IEEE,
not a multi-agent systems paper, not a “we beat GCN with TGN” paper.

---

## 2. Standing rules (non-negotiable)

1. **Do not invent new experiments** unless the user asks. Results are frozen.
2. **Do not quote live-API or LangGraph numbers as predictive skill.**
3. **Do not “discover” the overlap mechanism.** Cite Hansen–Hodrick, Richardson–Stock, Valkanov, Boudoukh–Richardson–Whitelaw. The contribution is import + measurement on this panel.
4. **Canonical numbers live in** `results/canonical_numbers.csv`. If a table in `paper.tex` disagrees with that CSV, the CSV is the source of truth **except** where the paper already documents seed-mean vs ensemble (see §7).
5. **Python for paper scripts:** `backend/.venv/bin/python` (fallback `dataset/.venv/bin/python`).
6. **Do not commit secrets.** `backend/.env` and `dataset/.env` exist locally.
7. **Do not claim a public data release** until a URL exists. Artifact docs are local (`artifact/`, `reproduce.sh`).
8. **Do not rewrite `paper.tex` back into a multi-agent / TGN-architecture paper.**
9. Prefer editing `paper.tex` over the stale `Research Paper/README.md` (old IEEE dump).

---

## 3. If you are asked to X, open Y

| User ask | Open first | Do not open first |
|---|---|---|
| Change a paper number, table, or claim | `results/canonical_numbers.csv`, `Research Paper/paper.tex`, this file §7 | `backend/`, `project2/`, live `/analyze` |
| Re-run paper tables (no new experiments) | `reproduce.sh`, `dataset/build_canonical.py` | `graph_convergence.py` unless asked (slow) |
| Target definitions T0/T1/T4/\(H_k\) | `dataset/phase0_t1_ladder.py` (`load()`), `dataset/horizon_sweep.py` | `dataset/training/data.py` (T0-style contraction for the old trainer) |
| Inference / DM / wild bootstrap | `dataset/dm_tests.py`, `dataset/ar1_decomposition.py` | `evidence/bootstrap_ci.py` (live-app CI, different question) |
| Graph models used **in the paper** | `dataset/graph_convergence.py`, `results/phase07/` | `backend/models/tgn_model.py` (serving wrapper + fallback) |
| Graph models used **in the old trainer** | `dataset/model_tgn.py`, `dataset/model_gcn.py`, `dataset/training/` | paper Phase 07 CSVs |
| Fused corpus / Comtrade vintage | `dataset/fuse_dataset.py`, `dataset/config.yaml`, `artifact/DATASHEET.md` | LangGraph news/weather agents |
| Live `/analyze` route | `backend/main.py`, `backend/orchestrator/` | paper results |
| Dashboard UI | `project2/src/` | `Research Paper/` |
| Deployment / API keys | `DEPLOYMENT_GUIDE.md`, `backend/.env.example` | paper |
| Lab chronology of Phase 0 | `changes.md` | treat numbers there as possibly superseded by the CSV |
| Reproduce figures | `figures/`, `dataset/make_paper_figures.py`, `dataset/per_origin_persistence.py` | `Research Paper/generate_results_figures.py` (older CLI figures) |

---

## 4. Repository map

```text
LAMDAAnalytics/
├── CLAUDE.md                 # short agent hook → this file
├── CODEBASE.md               # THIS FILE
├── README.md                 # live-app readme (ports/features); not the paper
├── changes.md                # Phase 0 lab log; can lag paper.tex
├── reproduce.sh              # rebuild canonical_numbers.csv from frozen graphs
│
├── Research Paper/           # TMLR manuscript (the scientific product)
│   ├── paper.tex             # CURRENT paper (anonymous TMLR)
│   ├── paper.bib / tmlr.sty / tmlr.bst / fancyhdr.sty
│   ├── paper.pdf             # compiled submission PDF
│   └── README.md             # STALE (old IEEE / multi-agent dump). Ignore for claims.
│
├── dataset/                  # corpus + paper experiments + old training stack
│   ├── config.yaml           # HS basket, India reporter 699, 2021–2024
│   ├── ingest_*.py           # download Comtrade / GDELT / POWER / GSCPI
│   ├── fuse_dataset.py       # causal monthly fusion + labels
│   ├── build_graph.py        # monthly country graph.json
│   ├── phase0_t1_ladder.py   # target family + tabular ladder (paper)
│   ├── build_canonical.py    # writes results/canonical_numbers.csv
│   ├── graph_convergence.py  # paper graph runs (80 ep, 5 seeds) — SLOW
│   ├── training/             # CLI trainer (train_models.py) — older protocol
│   └── data/four_year_2021_2024/
│       ├── processed/nodes_monthly.csv    # FUSED PANEL (864 rows, 18 countries)
│       └── results/model_training/        # archived trainer outputs (not paper tables)
│
├── results/                  # FROZEN paper numbers
│   ├── canonical_numbers.csv
│   ├── phase06/              # horizon sweep, origins, AR(1) decomp
│   └── phase07/              # graph seed means + ensemble preds
│
├── figures/                  # paper figures (overlap, origins, …)
├── artifact/                 # datasheet + anonymous-artifact notes
│
├── backend/                  # live FastAPI + LangGraph (NOT a paper contribution)
│   ├── main.py               # /analyze, /model/info, mock fallback
│   ├── orchestrator/         # StateGraph: geocode → 5 agents → normalize → tgn → report
│   └── models/tgn_model.py   # load tgn_model.pth or weighted_risk fallback
│
├── project2/                 # React + Vite dashboard (live app)
├── evidence/                 # timed live /analyze captures, not paper inference
├── start_backend.py          # uvicorn launcher
└── test_*.py                 # live-app / API smoke tests
```

Heavy / generated (do not “clean up” unless asked):
`dataset/data/four_year_2021_2024/cache/`, trainer `results/`, `evidence/runs/`,
`Research Paper/*.pdf` literature dumps, `backend/.venv/`.

---

## 5. Paper / corpus stack (the scientific product)

### 5.1 Panel

- File: `dataset/data/four_year_2021_2024/processed/nodes_monthly.csv`
- Unit: host country × month
- 18 countries, 48 months (2021-01 … 2024-12), **864 rows**
- `config.yaml` acquisition set is **20** countries; two drop out of the fused table
- Reporter: India, Comtrade code **699**, inbound (`flow_code = M`)
- HS-4 basket: `8517, 8471, 8528, 8516, 8507, 9403, 6109, 6203, 6204, 9503, 3304`
- Features on the row (seven-feature schema):
  `inventory_days_proxy`, `trade_delay_proxy`, `news_vol_7d`, `neg_tone_frac_3d`,
  `strike_flag_7d`, `weather_anomaly_7d`, `global_risk`
- Flow: `inbound_flow_usd` (vintage-filtered). Target columns on disk are T0-style;
  T1/T4/\(H_k\) are **re-derived in scripts**, not trusted as extra CSV columns.

Sources: UN Comtrade, GDELT, NASA POWER, NY Fed GSCPI. Datasheet: `artifact/DATASHEET.md`.

### 5.2 Split (forward-chained, used everywhere in the paper)

| Split | Months |
|---|---|
| Train | 2021-12 … 2022-12 |
| Validation | 2023 |
| Test | 2024 |

- T0 test **n = 193** country-months, **18** country clusters
- T1 test **n = 187**, **17** clusters (one country has no valid 2024 log return)
- Scalers / hypers / edge scales: **train only**

### 5.3 Target family (write these down; they are the paper)

Defined in `dataset/phase0_t1_ladder.py` → `load()` and `dataset/horizon_sweep.py`.

Let \(V_{v,t}\) be inbound flow.

| Id | Definition | Overlap |
|---|---|---|
| **T1** \(=H_1\) | \(\log(V_{T+1}/V_T)\) | 0 — **primary forecast target** |
| \(H_k\) | \(\log(V_{T+1}/V_{T-k+1})\) | \((k-1)/k\) |
| **T4** \(=H_{12}\) | \(\log(V_{T+1}/V_{T-11})\) | 11/12 (still overlapping) |
| **T0** | \((V_{T+1}-\mathrm{med}_{12})/\mathrm{med}_{12}\) | ≈11/12 of a slow denominator — **demonstration / inversion target**, not the forecast claim |
| T2 / T3 | denominator ablations defined in `target_definition_robustness.py`; canonical tuned-ridge rows written by `build_canonical.py` (paper Appendix B) | isolate which piece of T0 is the artifact |

Classification under T1 is **not reported** (10 positives). T0 \(\tau=0.20\) (20 positives, 10.4%) appears only as a parenthetical inversion demo.

### 5.4 Paper experiment scripts (dataset/)

Run from `dataset/` with the venv Python.

| Script | What it is | Output |
|---|---|---|
| `phase0_t1_ladder.py` | Naive + fitted tabular ladder on T0/T1/T4 | used by `build_canonical.py` |
| `naive_baselines.py` | Phase 0.1 persistence ladder (T0) | early gate |
| `target_definition_robustness.py` | T0–T3 overlap stress | early gate |
| `horizon_sweep.py` | \(H_k\) AC, overlap, persistence \(R^2\) | `results/phase06/horizon_sweep.csv` |
| `increment_memory.py` | T1 increment lag-2 / lag-3 AC | paper F1 residual-gap sentence |
| `ar1_decomposition.py` | univariate AR(1) vs AR(1)+flow; wild p vs 0 | `results/phase06/ar1_decomposition.csv` |
| `preprocessing_exogenous.py` | news / weather / GSCPI rebuild + ablation | canonical exo rows |
| `preprocessing_ablation.py` / `preprocessing_robustness.py` | older preprocessing studies | not the F3/F4 tables |
| `per_origin_persistence.py` | expanding-origin persistence | `results/phase06/per_origin_persistence.csv` |
| `dm_tests.py` | DM-HLN, pairs bootstrap, wild cluster | paper Table significance |
| `graph_convergence.py` | GCN / TGN / TGN-no-mem, 80 ep, patience 15, 5 seeds, T0 and T1 | `results/phase07/` |
| `graph_convergence_t1.py` / `graph_benchmark*.py` | earlier / partial graph runs | **superseded** by `graph_convergence.py` + phase07 |
| `build_canonical.py` | **single assembler** of `results/canonical_numbers.csv` | do not hand-edit the CSV |
| `make_paper_figures.py` | overlap / origin figures | `figures/` |
| `fuse_dataset.py` / `ingest_*.py` / `build_graph.py` | rebuild corpus from cache | only if user asks to re-acquire |
| `train_models.py` + `training/` | older matched-compute trainer | **not** the paper’s 80-ep 5-seed protocol |

`./reproduce.sh` at repo root = `build_canonical.py` only. It does **not** retrain graphs.

### 5.5 Old training package (`dataset/training/`)

This is a real, tested trainer (`dataset/tests/test_training.py`, `dataset/TRAINING.md`).
It is **not** what the TMLR tables are built from.

- Default target is T0-style contraction / \(\tau\) classification
- Default graph budget in `Options` is **10 epochs**, seed 7 — **not** the paper’s 80/15/5
- Pair-pooled path (`train_pair_models.py`, `training/pair_*.py`) is a different unit (directed pair-month)
- Archived outputs under `dataset/data/.../results/model_training/` must **not** be copied into `paper.tex`

Use this package only if the user asks to retrain the **application** models or to debug `train_models.py`.

---

## 6. Frozen paper numbers (do not “improve” these)

Source: `results/canonical_numbers.csv` plus `results/phase07/`.

### Mechanism (F1)

T1 lag-1 AC **−0.449**; T4 **+0.677**; T0 **+0.321**. Persistence origin-mean tracks overlap. T1 increment lag-2 AC **+0.044**, lag-3 **+0.107**.

### Rank inversion (F2)

| | T0 \(R^2\) | T1 \(R^2\) |
|---|---|---|
| Persistence (0 params) | **0.648** | **−1.909** |
| Univariate AR(1) | 0.567 | **0.171** |
| 3-feature AR ridge | 0.572 | 0.138 |

### Clean-target ladder (F3) — T1 test 2024

AR(1) **0.171** → +flow **0.150** → 3-feat ridge **0.138** → +weather **0.144** (not monotone; within noise) → +news **0.008** → +GSCPI **−0.023** → +all **−0.054**.

News rebuild (2026-09-12): GDELT features now cover all 18 countries (the four missing ones had blank ISO3
codes in `country_universe.csv`). Only news-dependent rows moved: 7-feature ridge T0/T1/T4 0.378/−0.032/−0.128
(was 0.392/−0.020/−0.052); +news T0/T1 0.338/0.008; +all T0/T1 0.464/−0.054. Pipeline order is
`ingest_gdelt.py` → `repair_processed_sources.py` (sets the monthly strike flag from CAMEO 143/144) → `fuse_dataset.py`.
Canonical CSV now stores 6 decimals to avoid double rounding.

AR(1) vs zero: wild **p = 0.001**, DM **p = 0.003**, **17 clusters**.
AR(1) vs TGN/GCN/RF/ridge: wild **p ≥ 0.22** (ties).

Phase 08 (review request, `dataset/seasonal_and_rolling.py` → `results/phase08/`):
AR(1,12) on T1 **0.215**; vs AR(1) wild **p = 0.037**, vs GCN ensemble **0.022**, ties with ridge/RF/TGNs
(wild 0.064–0.112), all unadjusted. Rolling origins (4 blocks, refit per origin): AR(1,12) best on T1 in
every block (mean 0.206; AR(1) 0.155, ridge 0.148, RF 0.099). On T0 the lag-12 term adds nothing (0.561).

### Graphs — two different \(R^2\)s (do not “fix” this by picking one)

| Model | T0 seed mean | T1 seed mean | **T1 ensemble** (DM uses this) |
|---|---|---|---|
| GCN | 0.569 ± 0.054 | 0.077 ± 0.025 | **0.119** |
| TGN | 0.539 ± 0.090 | 0.125 ± 0.024 | **0.133** |
| TGN no-mem | 0.599 ± 0.047 | 0.123 ± 0.016 | **0.130** |

Ensembling helps GCN most. Even ensemble GCN **does not** beat AR(1) 0.171.
Single-seed 40-epoch runs manufactured a “memory lift” and a “GCN inversion”; five-seed matched budget erased both. That is a reported finding, not a bug to reopen.

### Exogenous (F4)

News and GSCPI **hurt** on both T0 and T1. Weather is within noise. Canonical T0 AR baseline is **0.572** (not 0.577).

### Origins (F5)

T0 persistence swings **−0.169 … +0.708**. T1 stays roughly **−1.6 … −2.2**.

---

## 7. Manuscript

| File | Role |
|---|---|
| `Research Paper/paper.tex` | **The paper.** TMLR, anonymous, nine sections + appendix literature table |
| `Research Paper/paper.tex` `thebibliography` | Inline natbib bibliography (author–year). 20 keys between `ref2` and `ref34`; `paper.bib` is legacy and unused |
| `Research Paper/paper.pdf` | last compile (12 pp, US letter; Appendix A literature, Appendix B T2/T3) |
| `figures/overlap_vs_skill.pdf` | Fig F1 |
| `figures/per_origin_persistence.pdf` | Fig F5 |

Compile:

```bash
cd "Research Paper"
pdflatex paper && pdflatex paper   # inline bibliography, no bibtex
```

TeX Live may live at `~/.local/texlive/2026/bin/x86_64-linux`.

**Do not** restore IEEEtran or “we discovered overlap.”
Citation style is currently numeric `[n]` with references after the appendices, ordered by
first citation (author's choice, 2026-09-11). For TMLR submission, delete the
`\PassOptionsToPackage{numbers,...}{natbib}` and `\setcitestyle{numbers,...}` lines in
`paper.tex` to restore TMLR author–year.
Related-work table is **Appendix A** (`tab:litsynth`), not a two-column `table*`.

Data Availability must not claim a hosted URL until one exists.

---

## 8. Live application stack (not the paper)

### 8.1 Backend

- Entry: `backend/main.py` (FastAPI). Launcher: `start_backend.py` (uvicorn).
- Orchestrator: `backend/orchestrator/orchestrator.py` → `graph/network.py`
- Topology: `START → geocode → {trade, news, weather, political, gscpi} → normalize → tgn → report → END`
- Agents: `backend/orchestrator/agents/*.py`
- Shared state: `backend/orchestrator/graph/state.py`
- Request schema: `backend/orchestrator/utils/schema.py`
  (`component_type`, `seller_location`, `import_location`, …)
- Features assembled for TGN: inventory_days, past_delay_days, news_vol_7d,
  neg_tone_frac_3d, strike_flag_7d, weather_anomaly_7d, global_risk
- Serving TGN: `backend/models/tgn_model.py` loads `backend/tgn_model.pth` or
  falls back to `orchestrator/utils/scoring.py` `weighted_risk`
- **If APIs fail, `/analyze` returns a mock** (`main.py`). Demos can look “successful” without keys.
- Config: `backend/config/settings.py`, secrets in `backend/.env`

**Port mismatch (known):** root `README.md` says backend **8007** / frontend **5175**.
`project2/src/services/api.js` calls **`http://127.0.0.1:8000`**.
CORS in `main.py` allows 5173–5175. Check which port is actually bound before “fixing” CORS.

### 8.2 Frontend (`project2/`)

Vite + React. `src/App.jsx` → `Dashboard.jsx`, `SupplyChainMap.jsx`.
API wrapper: `src/services/api.js` (`/analyze`, `/model/info`, `/analytics/overview`, `/monitoring/alerts`).

`/analytics/overview` and `/model/info` in `main.py` currently return **hard-coded demo stats**, not corpus aggregates.

### 8.3 Evidence (`evidence/`)

Timed live `/analyze` runs, screenshots, bootstrap on **live scores**.
This is engineering evidence for the demo, **not** Diebold–Mariano on T1.

---

## 9. How data flows (do not reverse the arrows)

```text
Comtrade / GDELT / POWER / GSCPI
        │  ingest_*.py  (cache under dataset/data/.../cache/)
        ▼
 fuse_dataset.py  →  processed/nodes_monthly.csv
        │
        ├──────────────► phase0 / horizon / AR1 / exo / DM / graph_convergence
        │                         │
        │                         ▼
        │              results/canonical_numbers.csv  +  results/phase07/
        │                         │
        │                         ▼
        │                    paper.tex tables
        │
        └──────────────► train_models.py (old trainer) → dataset/.../results/model_training/
                                  │
                                  ▼
                         backend/tgn_model.pth  (serving; not a paper result)

Live path (separate):
  browser → POST /analyze → LangGraph agents (LLM/SERP/weather) → seven features
         → TGNWrapper or weighted fallback → JSON report
  Extractors are NOT validated against nodes_monthly.csv.
```

---

## 10. Environment cheat sheet

| Need | Where |
|---|---|
| Paper / sklearn / lightgbm | `backend/.venv/bin/python` |
| Dataset ingest keys | `dataset/.env` (`COMTRADE_*`, optional NOAA, GCP) |
| Live LLM / SERP / Mappls / weather | `backend/.env` |
| Frontend | `cd project2 && npm i && npm run dev` |
| Backend | `python start_backend.py` or uvicorn from `backend/` |
| Corpus rebuild | `dataset/README.md` (cache-first; do not hit Comtrade unless asked) |
| Trainer guide | `dataset/TRAINING.md` |

---

## 11. Documents that are stale or scoped

| File | Trust |
|---|---|
| `Research Paper/paper.tex` | **Current scientific claim** |
| `results/canonical_numbers.csv` | **Current numbers** |
| `artifact/DATASHEET.md` | Current corpus description |
| `changes.md` | Useful chronology; header still talks as if prose was not rewritten — **it was** |
| Root `README.md` | Live app only; still sells TGN F1 / multi-agent as the product |
| `Research Paper/README.md` | Old IEEE paper text. Do not copy numbers from it |
| `AI_AGENT_PIPELINE.md`, `ORCHESTRATOR_README.md` | Live-app docs |
| `dataset/README.md` | Corpus pipeline; still describes T0 as “the” target |
| `dataset/TRAINING.md` | Old trainer; 10-epoch defaults |
| `graph_benchmark*_results.txt` | Superseded by `results/phase07/` |

---

## 12. What “done” looks like for common agent jobs

**Paper edit:** change `paper.tex` only if the number is already in the CSV or the user asked for a new run. Compile PDF. Do not silently round differently from the CSV.

**New experiment (only if asked):** new script under `dataset/`, write under `results/`, then `build_canonical.py` if it belongs in the source-of-truth table, then paper. Never paste trainer `model_training/` metrics into the TMLR tables.

**Live-app bug:** stay in `backend/` + `project2/`. Do not retarget `paper.tex` around a UI fix.

**Do not:**
- Reopen graph seeds to “get TGN above AR(1)”
- Treat T0 \(R^2 \approx 0.65\) as a forecasting result
- Cite overlap bias as original
- Validate LangGraph extractors by pointing at `nodes_monthly.csv` without an actual alignment study
- Hand-edit `canonical_numbers.csv`
)

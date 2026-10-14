# LAMDA: supply-chain contraction forecasting

LAMDA is a country-month benchmark for forecasting contractions in a country's imports, together
with a multi-agent risk-scoring service and a React dashboard. It has three parts:

- **Dataset.** Monthly bilateral trade from UN Comtrade, fused with news events (GDELT), weather
  (NASA POWER) and the New York Fed Global Supply Chain Pressure Index (GSCPI). The features for
  month *T* are built only from month *T* and earlier.
- **Benchmark.** Tabular baselines (persistence, autoregressive, ridge, gradient boosting) and graph
  models (GCN, TGN), each scored on time-ordered splits and rolling origins. Every number has a script
  that reproduces it.
- **Service.** A FastAPI backend that runs the evaluated forecasting model (`/forecast`) and a
  LangGraph multi-agent pipeline that scores a supplier route with language-model agents (`/analyze`),
  plus a React dashboard.

## Main findings

The target is next month's change in a country's imports. The original panel has 18 countries; the
extended panels have 41 countries and two product baskets. Out-of-sample R² on the one-month log change:

| Evaluation | Own lags | Seven fused features | Graph models (GCN, TGN) | News, weather, GSCPI only |
|---|---|---|---|---|
| Original panel: 18 countries, semiconductors, 2021–2024, test 2024 (187 rows) | **0.215** | −0.250 | −0.089 to 0.055 | −0.260 |
| Extended panel: 41 countries, semiconductors and consumer goods, 2019–2025; ten half-year test blocks, 2021–2025 (4,738 rows) | **0.173** (positive in 10 of 10) | 0.108 | not run on blocks | −0.012 |
| Extended panel: 41 countries, both baskets, 2019–2025; test 2024–2025 (1,788 rows) | **0.185** | 0.098 | 0.094 to 0.118 | — |

- The target's own history is the best predictor in these evaluations. News, weather and GSCPI add at
  most about 0.001 R². No graph hyperparameter setting tested reaches own lags on the seven-year panel.
  One exception: on 41-country consumer goods in 2024 alone, the graph models score highest, but that
  lead is not significant and disappears over two test years.
- An overlapping-window contraction target, the usual choice, makes skill look far higher than it is.
  The zero-parameter persistence rule scores 0.648 on it and −1.909 on the non-overlapping log change.
- Unreported Comtrade months are treated as missing, not as zero imports. See
  [Unreported months](#unreported-months).

Per-result tables are in `results/paper21_revision/`. The checks behind each extended-panel number
are in `results/paper21_revision/NUMBERS_REVIEW.md`.

## Repository layout

| Path | Contents |
|---|---|
| `dataset/` | Data pipeline (`ingest_*.py`, `fuse_dataset.py`, `build_graph.py`), graph trainer (`training/`, `model_gcn.py`, `model_tgn.py`), configs, and every analysis script |
| `dataset/data/four_year_2021_2024/processed/` | Original panel: 18 countries, HS 8541/8542 (semiconductors), 2021-01 to 2024-12. `nodes_monthly.csv` is the country-month table; `edges_monthly.csv` and `graph.json` hold the bilateral flows; the rest are the source tables it was fused from |
| `dataset/data/four_year_2021_2024/processed_v2/` | Consumer-goods basket (11 HS codes), same 18 countries |
| `dataset/data/four_year_2021_2024/processed_v3/` | Both baskets, 41 countries, 2021-01 to 2024-12 |
| `dataset/data/four_year_2021_2024/processed_v4/` | Both baskets, 41 countries, 2019-01 to 2025-12 |
| `results/` | Outputs of the analysis scripts, including saved predictions of every graph-model run |
| `backend/` | FastAPI service, LangGraph agents, served model bundle (`models/serving/`), parity test |
| `project2/` | React dashboard (Vite) |
| `evidence/` | Orchestration and extraction studies, bootstrap intervals, CLI transcripts |
| `figures/` | Figures |
| `docs/` | Service documentation: API, agent pipeline, orchestrator, deployment; lab notes |
| `tests/` | Integration scripts for the service (most need API keys) |

## Setup

Python 3.12; a CPU is enough. `requirements.txt` pins the versions used for every result. The
analysis scripts expect the environment at `backend/.venv`.

```bash
git clone https://github.com/AdityaTakuli/LAMDAAnalytics.git
cd LAMDAAnalytics
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
backend/.venv/bin/pip install -r requirements.txt
```

## Reproducing the results

Run from `dataset/` with `PY=../backend/.venv/bin/python`. Each script prints its tables and writes
CSV files under `results/paper21_revision/`. Times are for one laptop CPU.

| Result | Command | Time |
|---|---|---|
| Benchmark on the original panel (unreported months treated as missing), tests, ablation, rolling origins | `$PY rev21_r2_corrected.py` | ~10 min |
| Every model with and without the unreported-month rows | `$PY rev21_r2_missing.py` | 1 min |
| Non-overlapping target, rolling origins, refit | `$PY rev21_tabular.py` | 1 min |
| Extended panels, 2021–2024 | `REV21_PANEL=processed_v3 REV21_OUT=phaseB $PY rev21_consumer_analyses.py` | minutes |
| Seven-year evaluation, ten half-year blocks | `$PY rev21_phaseD_analyses.py` | ~20 min |
| Graph models on the extended panels | `./rev21_run_graph_queue.sh`, then `$PY rev21_extended_graph_summary.py` | ~3 h |
| Graph hyperparameter sensitivity | `./rev21_run_graph_tune.sh`, then `$PY rev21_graph_tune_summary.py` | ~5 h |
| Other metrics, ridge strength, lag choice | `$PY rev21_extended_metrics.py` | ~15 min |
| Missing data, choice of countries and products, statistical assumptions | `REV21_WILD_REPS=1999 $PY rev21_r2_checks.py A` (and `B`, `C`) | ~1 h each for A and B |
| Bilateral diagnostics | `$PY rev21_bilateral.py` | minutes |

The graph queues skip runs that are already finished, so they can be restarted after an interruption.

## Running the service

### Backend

```bash
cd backend
cp .env.example .env        # add your keys; /forecast works without them
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Or run `python start_backend.py` from the repository root. API documentation is at
`http://127.0.0.1:8000/docs`.

| Endpoint | What it does |
|---|---|
| `GET /forecast?country=CHN&month=2024-06` | Next-month forecast from the evaluated models, on the published features. Matches the offline evaluation to machine precision |
| `POST /analyze` | LangGraph multi-agent route analysis (trade, news, weather, political, GSCPI agents → normalise → score → report) |
| `GET /model/info` | Information on the loaded scoring model |
| `GET /analytics/overview`, `GET /monitoring/alerts` | Dashboard data |

`/analyze` needs `GROQ_API_KEY`, `SERP_API_KEY`, `WEATHER_API_KEY` and `MAPPLS_API_KEY` in
`backend/.env`. Its scores come from language-model extraction and a fixed blend, not from the
evaluated forecasting model; use `/forecast` for the benchmarked numbers.

```bash
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d '{
  "component_type": "Semiconductor", "seller_location": "Hsinchu, Taiwan",
  "import_location": "Los Angeles, USA", "seller_name": "TSMC"}'
```

### Dashboard

```bash
cd project2
npm install
npm run dev                 # then open http://localhost:5173/dashboard
```

### Tests

```bash
cd backend && .venv/bin/python test_serving_parity.py                   # offline vs served forecast
cd backend && PYTHONPATH=. .venv/bin/python test_langgraph_network.py   # agent graph, mocked APIs
backend/.venv/bin/python tests/test_full_orchestrator.py                # live pipeline, needs keys
```

## Data

| Source | Used for |
|---|---|
| UN Comtrade | Monthly bilateral imports (HS 8541/8542; consumer basket of 11 HS-4 codes) |
| GDELT 1.0/2.0 | Monthly news volume, tone and strike counts per country |
| NASA POWER | Daily weather, aggregated to monthly anomalies |
| NY Fed GSCPI | Global supply-chain pressure |

The raw Comtrade replies (about 20 GB) and raw GDELT event files (about 24 GB) are not included. The
download and build scripts in `dataset/` rebuild them (`ingest_*.py` and `fuse_dataset.py` for the
original panel; `rev21_download_*.py` and `rev21_build_*_panel.py` for the extended ones). Comtrade
requests need a key in `dataset/.env` (see `dataset/.env.example`). Use of the Comtrade-derived tables
is subject to the UN Comtrade terms of use.

### Unreported months

Comtrade publishes a country's monthly trade only after the country submits it. Some reporter-months
were not published when the panels were built: Vietnam 2024 and 2025, China and Cambodia 2025,
Estonia 2019-12 and Philippines 2025-10. The panel files store these as zero imports. The analysis
scripts treat them as missing: a row whose current or next month is unpublished has no observable
target and is not scored.

## Team

- **Devansh Behl**: full-stack development
- **Mayan Sharma**: AI/ML engineering
- **Aditya Takuli**: data engineering and analytics
- **Lay Gupta**: product and business model

# India-centered supply-chain dataset (2021–2024)

Self-contained pipeline for acquiring, fusing, graph-building, analysis, and
model training on a single four-year country-month corpus.

```text
config.yaml  ->  data/four_year_2021_2024/
```

## Study design

* **Window:** January 2021 – December 2024 (48 months, 20 countries).
* **Trade:** India-centered e-commerce consumer-goods basket (HS-4 codes in
  `config.yaml`).
* **Graph:** country-level monthly snapshots; no firm-level edges invented from
  trade data.
* **Target:** one-month-ahead inbound contraction (continuous regression is the
  default; binary classification is gated when positives are too rare).

## Layout

```text
dataset/
├── config.yaml                    # single profile (2021–2024)
├── ingest_*.py                    # source downloaders
├── fuse_dataset.py                # causal monthly fusion + labels
├── build_graph.py                 # temporal graph serialization
├── analyze_four_year_data.py      # coverage / readiness report
├── audit_zero_inbound.py          # zero-trade audit (report-only)
├── train_models.py                # model training entry point
├── check_training_env.py          # preflight (packages, device, data)
├── TRAINING.md                    # full training guide
├── training/                      # training package
├── tests/test_training.py         # 26 training tests
└── data/
    └── four_year_2021_2024/
        ├── cache/                 # resumable source caches
        ├── processed/             # fused tables + graph.json
        └── results/
            ├── data_analysis/     # readiness report
            ├── model_training/    # final GCN/TGN regression run
            └── zero_value_audit.csv
```

## Setup

```bash
cd dataset
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt
cp .env.example .env
```

Set Comtrade keys in `.env` (`COMTRADE_SUBSCRIPTION_KEY`, `COMTRADE_SECONDARY_KEY`).

## Rebuild the corpus (optional)

The processed tables are already present. To rebuild from caches:

```bash
python ingest_comtrade.py --config config.yaml
python ingest_gdelt.py --config config.yaml
python ingest_weather.py --config config.yaml
python ingest_gscpi.py --config config.yaml
python fuse_dataset.py --config config.yaml
python build_graph.py --config config.yaml
```

## Analysis

```bash
python analyze_four_year_data.py --config config.yaml
python audit_zero_inbound.py
```

## Training (final results)

Preflight and train:

```bash
python check_training_env.py --config config.yaml
python train_models.py --self-test --device cpu
python train_models.py --config config.yaml --task regression --device cpu
```

Final regression artifacts (already produced):

```text
data/four_year_2021_2024/results/model_training/regression/
├── run_summary.json
├── metrics/comparison.csv
├── checkpoints/
├── diagnostics/
└── plots/
```

See `TRAINING.md` for every CLI option, metric interpretation, and checkpoint
loading.

## Chronological split (model training)

| Partition | Months | Role |
| --- | --- | --- |
| Train | 2021-12 … 2022-12 | fit features, baselines, graph models |
| Validation | 2023 | model selection |
| Test | 2024-01 … 2024-11 | held-out scoring (2024-12 excluded: no Jan 2025 target) |

The first eleven months of 2021 are dropped from training because a strict
12-month baseline median requires a full history.

## Headline test results (2024 hold-out)

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Ridge | 0.182 | 0.142 | 0.410 |
| TGN (no memory) | 0.209 | 0.145 | 0.218 |
| GCN | 0.215 | 0.146 | 0.176 |
| TGN | 0.229 | 0.172 | 0.061 |
| Train median | 0.251 | 0.164 | -0.122 |

Ridge regression beat the graph models on this split. Full metrics are in
`metrics/comparison.csv`.

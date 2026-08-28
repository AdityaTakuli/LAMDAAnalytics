# India-Centered E-Commerce Supply-Chain Pipeline

Self-contained data acquisition, fusion, graph, and TGN/GCN training code.
The pipeline is staged:

```text
config.yaml          -> 2024 one-year corpus
config_4year.yaml    -> 2021-2024 four-year expansion
```

Both profiles write to separate directories under `dataset/data/`.

## Study design

India is the fixed focal reporter. The other countries are selected from
India's observed ECGB import partners by configured-period trade value,
retaining the top 19 partners for a 20-country graph. The trade basket
is:

```text
8517  smartphones and telecom equipment
8471  computers and computing equipment
8528  televisions, monitors, and displays
8516  electrical household appliances
8507  batteries and accumulators
9403  furniture
6109  t-shirts and vests
6203  men's and boys' apparel
6204  women's and girls' apparel
9503  toys
3304  cosmetics and beauty products
```

The graph is country-level. It does not claim to observe Amazon, Walmart, or
Flipkart firm telemetry and does not invent firm edges from country-level
trade. `ingest_cset.py` remains as an isolated legacy semiconductor utility
but is not part of this e-commerce pipeline.

## Setup

```bash
cd dataset
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

```bash
# Required for a complete authenticated Comtrade pull
COMTRADE_SUBSCRIPTION_KEY=...
COMTRADE_SECONDARY_KEY=...

# Optional legacy BigQuery settings; the reproducible downloader uses
# official daily raw exports and does not require these credentials.
GCP_PROJECT=...
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/service-account.json

# Required only for weather.provider=noaa
NOAA_CDO_TOKEN=...
```

The keys are loaded only from `.env` and are never stored in YAML. A
Comtrade key is required for the full India/all-partner corpus; the no-key
preview endpoint is record-limited.

## Stage 1: one year

The default `config.yaml` covers **January–December 2024** and writes to:

```text
dataset/data/one_year_2024/
```

Run from `dataset/`:

```bash
python ingest_comtrade.py --config config.yaml
python ingest_gdelt.py --config config.yaml
python ingest_weather.py --config config.yaml
python ingest_gscpi.py --config config.yaml
python fuse_dataset.py --config config.yaml
python build_graph.py --config config.yaml
python train.py --config config.yaml
```

`ingest_comtrade.py` creates
`data/one_year_2024/processed/country_universe.csv`. Run it before GDELT and
weather so both can restrict their data to the selected countries.

## Stage 2: expand to four years

After reviewing the one-year pull, run the separate expansion profile:

```bash
python ingest_comtrade.py --config config_4year.yaml
python ingest_gdelt.py --config config_4year.yaml
python ingest_weather.py --config config_4year.yaml
python ingest_gscpi.py --config config_4year.yaml
python fuse_dataset.py --config config_4year.yaml
python build_graph.py --config config_4year.yaml
python train.py --config config_4year.yaml
```

The four-year profile covers **January 2021–December 2024** and writes only
to:

```text
dataset/data/four_year_2021_2024/
```

This is a fresh four-year pull, not an in-place mutation of the one-year
directory. The per-source caches and derived outputs remain isolated.

For the GDELT-only smoke test, run:

```bash
python ingest_gdelt.py --config config.yaml --gdelt-test
```

The test defaults to **2020-01-01 through 2020-01-03** and writes under the
profile's `processed/test/` directory without changing the full output.
For the full five-year pull, omit `--gdelt-test`:

```bash
python ingest_gdelt.py --config config_4year.yaml
```

## Sources

1. **UN Comtrade** — India monthly imports, all partners initially, ECGB
   HS-4 codes, with trade value, volume, and vintage metadata.
2. **GDELT Event Database** — official daily event exports, actor countries,
   CAMEO event codes, tone, URLs, and a deterministic labor-unrest proxy.
   The configured GDELT window is January 2020–December 2024, independent of
   the graph profile's current 2024 or 2021–2024 analysis window.
3. **NASA POWER** — daily temperature for selected country centroids.
   NOAA CDO is optional.
4. **NY Fed GSCPI** — direct published Excel series; no LLM estimation.

Every source caches raw pulls under its profile's `data/.../cache/` directory.
GDELT uses the shared `dataset/cache/gdelt/YYYY/` directory so the same raw
days are reusable across profiles. It uses atomic `.tmp` downloads,
ZIP validation, exponential-backoff retries, daily processed partitions, and
`failed_downloads.json`. Existing valid ZIPs and processed partitions are
reused unless `--force` is passed.

Missing raw files download with a bounded `ThreadPoolExecutor`. The default
is `download_workers: 8`; the CLI `--workers` override accepts `1` through
`16` and never parallelizes feature processing. Use `--limit N` for a bounded
download/processing smoke test:

```bash
python ingest_gdelt.py --config config.yaml --workers 4 --limit 5
```

Each worker writes a temporary file, validates ZIP integrity, and atomically
renames it to the existing date-based filename. Failed files are retried with
exponential backoff and recorded in `cache/gdelt/failed_downloads.json`.

## Fused schema

`processed/nodes_monthly.csv` contains one country row per month:

* `timestamp`, `month`, `node_id`, `node_type`, `host_country_id`
* `inventory_days_proxy`, `trade_delay_proxy`
* `news_vol_7d`, `neg_tone_frac_3d`, `strike_flag_7d`
* `weather_anomaly_7d`, `global_risk`
* `inbound_flow_usd`
* `label`, `label_tau_0.30`, `label_tau_0.35`, `label_tau_0.40`
* `feature_provenance`, `feature_sources`, `is_inherited`, `vintage_date`

The first two features are explicitly Comtrade-derived proxies; Comtrade does
not directly observe platform inventory days or delivery delays.

`processed/edges_monthly.csv` contains country-to-country observed monthly
Comtrade import edges and preserves:

* `trade_value_usd`, `flow_volume`
* `edge_type`, `provenance`
* `is_observed`, `is_inherited`

No country flow is disaggregated to firms.

### GDELT event and feature definitions

The official URL format is:

```text
https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
```

Only selected ISO3 actor countries are retained. A country is assigned when
it occurs in Actor1CountryCode or Actor2CountryCode; events are not broadcast
to every country. If Comtrade has already produced `country_universe.csv`,
that universe takes precedence over the configured fallback list.

The parser retains the event identifier, SQLDATE/reference date, DATEADDED
availability timestamp, both actor countries, EventCode, EventBaseCode,
EventRootCode, AvgTone, SOURCEURL, country association, classification flags,
source-file date, source, and vintage date. GDELT's `DATEADDED` is a database
availability/coding timestamp, not a claim about the article's publication
time.

The default supply-chain event roots are CAMEO `14`, `16`, `17`, `18`, and
`19`. This is an explicit event filter, not a claim that every country event
is a disruption. Labor unrest is the narrower configurable CAMEO root `14`
(Protest) proxy. It is reproducible but can include non-labor protests and
cannot identify every real-world strike.

`gdelt_features.csv` is a complete country-by-day grid:

* `news_vol_7d`: count of supply-chain-relevant event-country rows by GDELT
  `DATEADDED` in the inclusive trailing seven calendar days.
* `neg_tone_frac_3d`: fraction of relevant rows by `DATEADDED` in the
  inclusive trailing three days with `AvgTone <= negative_tone_threshold`
  (default `-2.0`).
* `strike_flag_7d`: one if any configured labor-unrest event by `DATEADDED`
  occurs in the inclusive trailing seven days, otherwise zero.

The monthly table uses the UTC month-end daily row, preserving the trailing
window semantics rather than averaging window values. Event/reference dates
and availability timestamps must both be no later than the feature date, so
future events cannot enter historical features. `fuse_dataset.py` consumes
the monthly table when present and retains its existing event-level fallback.

## Causal alignment and labels

For monthly step `T`, the anchor is the last UTC day of the month. Features
must have both observation/reference date and publication/availability time
no later than the anchor.

* GDELT volume and labor unrest: trailing `[T-6 days, T]`
* GDELT negative tone: trailing `[T-2 days, T]`
* Weather: seven eligible daily observations, Eq. 6 anomaly rule
* GSCPI: latest eligible published value
* Comtrade: India inbound monthly flow

Labels use only inbound Comtrade value:

```text
V = inbound ECGB import value
y[T] = 1 if (V[T+h] - median(V[T+h-12:T+h-1])) /
            median(V[T+h-12:T+h-1]) < -tau
```

The horizon is one month and `tau` is swept over `0.30`, `0.35`, and `0.40`.
Strike and weather are features only. Rows without a valid future value or
positive baseline are unlabeled.

## GCN versus TGN

`model_gcn.py` implements a same-month snapshot GCN with mean neighbor
aggregation and no temporal memory.

`model_tgn.py` implements:

* learned harmonic time encoding;
* endpoint message MLP;
* mean message aggregation;
* GRU memory;
* recent-neighbor temporal attention;
* sigmoid node-risk decoder;
* linear fallback when a checkpoint is incompatible.

`train.py` compares:

```text
constant prevalence
hand-weighted linear
logistic regression
GCN
TGN
TGN-no-memory
```

## Training

The four-year profile uses:

* Train: 2021–2022, 24 months
* Validation: 2023, 12 months
* Test: 2024, 12 months

Events are processed chronologically without shuffling. Standardization and
edge scaling are fit on the training partition only. TGN memory resets at
fold boundaries, and weighted BCE uses inverse positive prevalence.

With a small country-level sample, metrics are directional rather than
conclusive. Prefer ROC-AUC, average precision, precision, recall, and lead
time over accuracy. Historical GDELT and a deployed SERP/scrape news agent
also have train/serve distribution differences that must be checked on a
common held-out period.

## Offline smoke test

The smoke path makes no network calls:

```bash
python fuse_dataset.py --config config.yaml --synthetic
python build_graph.py --config config.yaml
python train.py --config config.yaml --synthetic
```

Synthetic outputs are validation artifacts, not real empirical data.

## Layout

```text
dataset/
├── config.yaml
├── config_4year.yaml
├── ingest_comtrade.py
├── ingest_cset.py                 # legacy, not used by e-commerce graph
├── ingest_gdelt.py
├── ingest_weather.py
├── ingest_gscpi.py
├── fuse_dataset.py
├── build_graph.py
├── model_gcn.py
├── model_tgn.py
├── train.py
├── common.py
├── requirements.txt
└── data/
    ├── one_year_2024/
    └── four_year_2021_2024/
```

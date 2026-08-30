# India-Centered E-Commerce Supply-Chain Pipeline

Self-contained data acquisition, fusion, graph, and TGN/GCN training code.
The pipeline is staged:

```text
config.yaml          -> 2024 one-year corpus
config_4year.yaml    -> 2021-2024 four-year expansion
```

Both profiles write to separate directories under `dataset/data/`.

## Study design

India is the fixed focal country. The existing 20-country universe is
preserved from the legacy India import pull. The 2024 bilateral coverage-fix
profile uses these semiconductor HS-4 codes:

```text
8541  semiconductor devices
8542  electronic integrated circuits
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

### 2024 bilateral Comtrade coverage fix

The original one-year artifact is preserved before replacing the processed
Comtrade table:

```text
data/one_year_2024/processed/comtrade_coverage_fix_backup/
```

The coverage-fix command uses the existing `country_universe.csv` exactly as
its reporter and partner universe. It requests only 2024 monthly imports for
HS 8541 and HS 8542, with one resumable cached response per reporter-month:

```bash
python ingest_comtrade_bilateral.py --config config.yaml
```

Responses are cached under
`data/one_year_2024/cache/comtrade/bilateral/`. The command retries transient
HTTP/network failures with exponential backoff, respects
`request_delay_seconds`, records a manifest and failures, and never converts a
missing API cell into zero trade. It rebuilds `comtrade.csv`, the causal fused
country-month tables, and `graph.json` only after all requests succeed. It
also writes the reporter-partner-month coverage grid and target coverage
diagnostics under:

```text
data/one_year_2024/results/comtrade_coverage_fix/
```

This is a 2024-only coverage operation. It does not train GCN, TGN,
TGN-no-memory, or logistic regression.

## Independent daily diagnostic profile

`config_daily.yaml` and `build_daily_pipeline.py` are a separate 2024-only
profile. They write only to:

```text
dataset/data/one_year_2024_daily/
```

The daily profile reuses the existing 2024 GDELT daily features and NASA
POWER daily observations. It carries monthly Comtrade structural features and
NY Fed GSCPI values within their calendar month, while marking their original
frequency and the explicit `month_start` availability assumption. Existing
monthly Comtrade edges are used as monthly edge templates for each daily
snapshot; no daily trade values or daily topology changes are fabricated.

Run the diagnostic stage with:

```bash
python build_daily_pipeline.py --config config_daily.yaml
```

The command creates 366 daily snapshots and 20 country-day rows per date,
then writes target diagnostics and a leakage audit. The current public 2024
sources do not contain an independent daily disruption ground truth, so the
command stops before model training. It creates no daily checkpoints,
predictions, or metrics. Daily outputs never overwrite the monthly profile.

The isolated daily-training gate can be run separately:

```bash
python train_daily.py --config config_daily_training.yaml
```

It writes only to `data/one_year_2024_daily_training/`. It records the
chronological Jan–Jun / Jul–Sep / Oct–Dec 30 split and copies the diagnostic
evidence, but currently blocks GCN/TGN training because the daily target is
not scientifically established.

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

### Acquisition-only 2021–2023 download

To acquire only the three additional years without touching either 2024
profile, use:

```bash
python download_three_years.py --config config_three_year_download.yaml
```

This writes raw GDELT ZIPs, NASA POWER JSON responses, bilateral Comtrade
JSON responses, and the NY Fed workbook only under
`data/three_year_2021_2023/`. It does not fuse tables, build graphs, or train
models. The resumable manifest at
`data/three_year_2021_2023/download_manifest.json` records successful,
unavailable, and authorization-failed requests. CSET is static and is not
redownloaded as a year-specific source.

#### Current three-year acquisition status

The missing Comtrade requests are retried one year at a time with:

```bash
python download_comtrade_year.py --config config_three_year_download.yaml --year 2021
python download_comtrade_year.py --config config_three_year_download.yaml --year 2022
python download_comtrade_year.py --config config_three_year_download.yaml --year 2023
```

The current status is **incomplete**:

* 2021: 240 of 240 Comtrade reporter-month requests succeeded after a later
  retry.
* 2022: 168 of 240 succeeded; 72 failed.
* 2023: 164 of 240 succeeded; 76 failed.
* Total Comtrade: 572 of 720 succeeded; 148 remain failed.
* The 2022 failures and most 2023 failures are HTTP 403; four 2023 retries
  also encountered temporary DNS failures.
* GDELT: 1,093 of 1,095 daily files succeeded; 2022-11-10 and 2023-03-23
  returned HTTP 404 because the official daily exports are unavailable.
* Weather: all 20 configured location files succeeded.
* GSCPI: the published workbook succeeded.

The exact cause of an HTTP 403 must be confirmed in the Comtrade account
dashboard, but the observed pattern strongly suggests a request/quota limit:
the initial run stopped after roughly 500 successful calls, and later
year-specific retries succeeded after the quota window changed. A 403 is a
rejected request, not an observed zero-trade value. Splitting requests by
year is useful for resumability, but it cannot bypass an account quota. The
year-specific manifests are the current source of truth:

```text
data/three_year_2021_2023/comtrade_2021_manifest.json
data/three_year_2021_2023/comtrade_2022_manifest.json
data/three_year_2021_2023/comtrade_2023_manifest.json
```

Do not describe this directory as a complete 2021–2023 dataset until the
Comtrade credentials/quota are fixed and all required requests have either
succeeded or been scientifically excluded. The original aggregate
`download_manifest.json` is an initial three-year snapshot and may still show
the earlier 500/220 result; inspect the three year-specific manifests after a
retry. The two GDELT 404 dates should remain documented as unavailable source
files rather than fabricated empty files.

#### Three-year directory and file structure

The acquisition-only profile does not create processed tables. Its layout is:

```text
data/three_year_2021_2023/
├── download_manifest.json
├── comtrade_2021_manifest.json
├── comtrade_2022_manifest.json
├── comtrade_2023_manifest.json
└── cache/
    ├── comtrade/
    │   └── bilateral/
    │       └── reporter_<code>_<YYYYMM>.json
    ├── gdelt/
    │   └── <YYYYMMDD>.export.CSV.zip
    ├── weather/
    │   └── <location>.json
    └── gscpi/
        └── gscpi_data.xlsx
```

There are 20 configured reporters, 20 configured partners, and 12 months per
year. The Comtrade request unit is **one reporter and one month**, with all
20 partners and HS 8541/8542 requested in that response:

```text
3 years × 12 months × 20 reporters = 720 requests
```

The `reporter_<code>_<YYYYMM>.json` wrapper contains:

```text
request.reporter_code
request.partner_codes
request.period
request.flow_code = "M"
request.commodity_codes = ["8541", "8542"]
response.count
response.data[]
response.error
downloaded_at
```

Each `response.data[]` row is an official Comtrade observation. Important
fields include `reporterCode`, `reporterISO`, `partnerCode`, `partnerISO`,
`flowCode`, `period`, `cmdCode`, `primaryValue`, `netWgt`, and `qty`.
The exact field set is supplied by Comtrade and can vary by response. A
successful response with an empty `data` list is an observed no-record
response; a failed request is **unknown**, not zero trade.

The GDELT cache contains official daily ZIP exports. The weather cache contains
NASA POWER daily JSON observations for the configured country centroids. The
GSCPI cache contains the direct NY Fed Excel workbook. These files are raw
inputs and should be parsed into separate processed tables before modeling.

The acquisition directory is intentionally not yet an ML-ready dataset:
`processed/`, `results/`, and graph artifacts are absent there. This prevents a
partial Comtrade cache from silently becoming a model's training set.

#### How the team should use this data for ML

Do not train directly from the raw JSON, ZIP, or Excel files. The intended
workflow is:

```text
raw cache
  -> source validation and country-code mapping
  -> processed source tables
  -> causal country-month fusion
  -> temporal graph snapshots
  -> target construction
  -> chronological train/validation/test split
  -> model training and evaluation
```

The recommended supervised unit is one **country-month** row. For each month
`T`, keep the feature timestamp at or before the month end and predict the
next-month inbound Comtrade contraction:

```text
X(country, T) -> y(country, T+1)

future_value = inbound_flow_usd(country, T+1)
baseline = median(inbound_flow_usd(country, T-11) ... inbound_flow_usd(country, T))
contraction = (future_value - baseline) / baseline
label = 1 when contraction < -tau
```

Use a fixed, pre-specified `tau` such as 0.30, or report the planned
0.30/0.35/0.40 sensitivity sweep. Do not select `tau` using test
performance. Rows with missing future Comtrade data, missing baseline, or a
non-positive baseline must have `label_valid = 0`; they must not be silently
converted to negative examples.

For a completed combined 2021–2024 study, a defensible first split is:

```text
Train:      2021-01 through 2022-12
Validation: 2023-01 through 2023-12
Test:       2024-01 through 2024-12
```

The first 12 months cannot use a 12-month historical baseline, and the final
month cannot use a next-month target unless January 2025 is acquired. The
split must be applied by calendar month, with no random shuffling. Fit
standardization, imputation rules, edge scaling, and any feature selection on
the training period only.

The ML team should not start the combined 2021–2024 experiment from the
current partial cache. Missing Comtrade reporter-month responses are unknown
labels/features, not negative examples. First complete the required cache,
run a separate processing profile, audit coverage and class balance, and only
then expose the resulting `nodes_monthly.csv`, `edges_monthly.csv`, and
`graph.json` to `train.py`. The existing 2024 monthly and daily experiments
remain separate and are not replaced by this acquisition.

Recommended handoff checklist:

```text
[ ] Comtrade manifests show the required reporter-month requests complete
[ ] Failed GDELT dates are retained in the provenance report
[ ] Country codes are mapped consistently to the 20-country universe
[ ] No failed API request is filled with zero
[ ] Processed rows and bilateral edges pass coverage checks
[ ] Label-valid rows have a positive historical baseline and a known future value
[ ] Train/validation/test months are chronological and disjoint
[ ] Scalers and feature selection are fit on training months only
[ ] Training contains both positive and negative classes
[ ] Baselines are run before GCN/TGN comparisons
```

The model inputs should remain country-level:

* Comtrade-derived structural features: inbound flow, trade-value changes,
  `inventory_days_proxy`, and `trade_delay_proxy`.
* GDELT daily features aligned causally to the month end:
  `news_vol_7d`, `neg_tone_frac_3d`, and `strike_flag_7d`.
* Weather daily observations aggregated using only observations available by
  the feature timestamp, including `weather_anomaly_7d`.
* NY Fed GSCPI monthly value aligned to its publication/availability date as
  `global_risk`.
* Bilateral Comtrade import edges for the country graph.

Do not claim these features measure Amazon, Walmart, or Flipkart inventory or
delivery systems. They are public country-level trade and external-risk
proxies. Do not create firm-level edges from country-level trade, and do not
use `strike_flag` as the label because that would make the target tautological.

Before training, generate and inspect:

```text
processed/nodes_monthly.csv
processed/edges_monthly.csv
processed/graph.json
processed/label_summary.json
results/diagnostics/leakage_audit.json
```

The label summary must report total rows, valid rows, positive rows, negative
rows, prevalence, and missingness by country and month. If the training split
contains only one class, stop and report that limitation; do not rebalance by
fabricating labels or by changing the threshold after seeing model results.

For a first implementation, use the existing `train.py` as the reference
runner and compare the constant-prevalence baseline, logistic regression,
GCN, TGN, and TGN-no-memory under identical splits. Report accuracy,
balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, positive/negative
counts, and prevalence. Mark undefined one-class metrics as `N/A`. Reset TGN
memory only at the beginning of a chronological replay, never before every
individual month or evaluation row.

To complete the missing Comtrade data, configure valid keys in `.env`, verify
their API product and remaining quota with Comtrade, then rerun the
year-by-year command above. After the cache is complete, build a separate
multi-year processing profile; do not overwrite
`data/one_year_2024/` or `data/one_year_2024_daily/`.

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

1. **UN Comtrade** — selected-country bilateral monthly imports for HS 8541
   and HS 8542, with trade value, quantity/weight, and vintage metadata.
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

For the real one-year run, training artifacts are isolated under
`data/one_year_2024/results/`: model checkpoints, `training_metrics.json`,
`prediction_scores.json`, and a copy of the serialized graph. Raw source
caches, event files, and processed source tables remain in their original
locations for later multi-year expansion.

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

### Isolated 2024 country-month experiment

After the real 2024 graph has been built, run the separate experiment without
overwriting the prior training artifacts:

```bash
python country_month_experiment.py --config config.yaml
```

It evaluates the existing one-month-ahead inbound-flow contraction label at
the configured default `tau` (not selected from model performance), using the
chronological split January–July train, August–September validation, and
October–November test. December is excluded because its next-month target is
outside the 2024 dataset. The runner preserves all 20 country nodes per
snapshot, but only evaluates target-valid country-month rows. With the current
India-reporter Comtrade pull, partner nodes have no observed inbound target and
are therefore not treated as negative examples.

Artifacts are written to
`data/one_year_2024/results/country_month_experiment/`:

```text
diagnostics/       label distributions, split statistics, leakage audit
predictions/       per-country-month model scores
metrics/           per-model JSON and comparison.csv
checkpoints/       GCN, TGN, and TGN-no-memory checkpoints
graphs/            target, loss, comparison, and score plots
```

When a partition contains one class, discrimination metrics are recorded as
`N/A`; accuracy and balanced accuracy are descriptive only. This command uses
the existing processed 2024 tables and makes no network calls or synthetic
data.

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
├── ingest_comtrade_bilateral.py
├── ingest_cset.py                 # legacy, not used by e-commerce graph
├── ingest_gdelt.py
├── ingest_weather.py
├── ingest_gscpi.py
├── config_three_year_download.yaml
├── download_three_years.py
├── download_comtrade_year.py
├── fuse_dataset.py
├── build_graph.py
├── model_gcn.py
├── model_tgn.py
├── train.py
├── common.py
├── requirements.txt
└── data/
    ├── one_year_2024/
    ├── one_year_2024_daily/
    ├── one_year_2024_daily_training/
    ├── three_year_2021_2023/
    └── four_year_2021_2024/
```

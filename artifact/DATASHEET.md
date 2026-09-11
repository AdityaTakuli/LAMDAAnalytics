# LAMDA trade-disruption corpus — datasheet

Anonymized description of the fused historical corpus used in the paper.
Do not treat this file as a public-release URL. Link this archive at TMLR
submission; until then the paper does not claim a released dataset.

## Motivation

A public country-month panel for measuring how overlapping-window disruption
targets manufacture forecasting skill, and for testing whether exogenous
channels and graph architectures beat a univariate AR(1) once overlap is
removed.

## Composition

| Field | Value |
|---|---|
| Unit | host country × month |
| Span | 2021-01 through 2024-12 (48 months) |
| Analysis panel | 18 countries, 864 rows (`nodes_monthly.csv`) |
| Acquisition set | 20 countries in `dataset/config.yaml`; two countries drop out of the fused analysis table |
| Reporter | India (UN Comtrade reporter code 699), inbound flow (`flow_code = M`) |
| HS basket (HS-4) | 8517, 8471, 8528, 8516, 8507, 9403, 6109, 6203, 6204, 9503, 3304 (India-centered e-commerce consumer-goods basket) |
| File | `dataset/data/four_year_2021_2024/processed/nodes_monthly.csv` |

Test sizes used in the paper: **n = 193** country-months with a valid T0 label
in 2024; **n = 187** with a valid T1 log return. Cluster counts: **18** under
T0, **17** under T1.

## Sources and licenses

Reuse is subject to each provider's terms. This archive redistributes only the
fused research table, not raw provider dumps.

| Channel | Provider | Role | License / terms (summary) |
|---|---|---|---|
| Inbound trade value | UN Comtrade | \(V_t\), vintage-filtered | UN Comtrade terms of use; attribution required |
| News volume and tone | GDELT | `news_vol_7d`, `neg_tone_frac_3d` | GDELT data may be used with attribution |
| Temperature | NASA POWER | weather anomaly / seasonal \(z\) | NASA public data |
| Global stress | NY Fed GSCPI | `global_risk` and GSCPI dynamics | Federal Reserve Bank of New York, public research series |

A timestamp bug in an earlier fusion vintage inflated ridge \(R^2\) from 0.349
to 0.410. All paper numbers use the repaired fusion.

## Target family

Let \(V_{v,t}\) be inbound flow of country \(v\) in month \(t\).

| Id | Definition | Overlap with own lag |
|---|---|---|
| T1 \(= H_1\) | \(\log(V_{T+1}/V_T)\) | 0 |
| \(H_k\) | \(\log(V_{T+1}/V_{T-k+1})\) | \((k-1)/k\) |
| T4 \(= H_{12}\) | \(\log(V_{T+1}/V_{T-11})\) | 11/12 |
| T0 | \((V_{T+1}-\mathrm{med}_{12})/\mathrm{med}_{12}\) | \(\approx 11/12\) of the denominator |

Targets are constructed in `dataset/phase0_t1_ladder.py` (`load()`). Horizon
sweep: `dataset/horizon_sweep.py`.

## Split (forward-chained)

- Train: 2021-12 through 2022-12
- Validation: 2023
- Test: 2024

All scalers, hyperparameters, and graph edge scales are fit on training months
only. Vintage filters keep 2024 test months from seeing later Comtrade
revisions.

## Canonical numbers

`results/canonical_numbers.csv` is the single source of truth for every table
in the paper. Graph seed runs: `results/phase07/graph_convergence_{T0,T1}.csv`.
Seed-averaged predictions used in Diebold–Mariano tests:
`results/phase07/graph_preds_{T0,T1}.csv`.

## Collection

Fused from public APIs and files (`dataset/ingest_*.py`, `dataset/fuse_dataset.py`).
No personally identifying information. Countries are the unit; no firm names.

## Uses and distribution

Intended use: reproduce the paper's target-family, baseline, and inference
results. Not a live disruption feed. The LangGraph extractors mentioned in the
deployment note are unvalidated and are not part of this corpus.

## Reproduce

From the repository root:

```bash
./reproduce.sh
```

See `artifact/README.md`.

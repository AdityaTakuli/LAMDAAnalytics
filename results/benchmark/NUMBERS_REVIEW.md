# Data extension: numbers for review (consumer basket and extended panels)

> **Round-2 correction (2026-10-14).** Comtrade has no monthly data for Vietnam 2024 (and Vietnam/China/
> Cambodia 2025, Estonia 2019-12, Philippines 2025-10). Every panel stores such months as zero inbound value.
> On the overlapping target T0 a row whose next month is unreported gets c = -1 (fake collapse, label 1).
> Original panel, test 2024: 6 of 193 rows (52% of the target variance, 6 of 20 positives). Without them:
> ridge 0.338 -> -0.242, GCN 0.328 -> -0.080, TGN 0.308 -> -0.188, TGN-nm 0.314 -> -0.111, persistence
> 0.648 -> 0.247, own lag 0.558 -> 0.364, trade-only 0.394 -> 0.021, best PR-AUC 0.399 -> 0.097 (prevalence
> 0.104 -> 0.075). All T1 numbers below are unaffected (T1 is undefined when either month is zero). The T0
> numbers in section 5 and the T0 graph rows in section 6 include the artefact rows; corrected values are
> in `corrected/corrected_extended_T0.csv` and `corrected/corrected_extended_graph_T0.csv`
> (e.g. consumer 41, ten blocks: own lag 0.431 -> 0.408; 41-country consumer graph T0: AR 0.345 -> 0.191,
> graphs 0.095-0.166 -> -0.082 to -0.029). Statistical-assumption checks: `corrected/checks_statistics.csv`
> (AR(1,12) vs all seven, both baskets: two-way clustered p = 0.073; Holm 0.036; block bootstrap 0.003;
> sign test 9/10, p = 0.021).

Audit date 2026-10-06. Nothing here is in the paper or the response letter yet.
All R² are out-of-sample. T1 = one-step log change `log(V[T+1]/V[T])` (no window overlap).
T0 = the paper's rolling-median contraction. "Own lags" = AR(1,12) on the target's lags 1 and 12.

## 1. What was checked

| Check | Result |
|---|---|
| Duplicate records in each download batch (consumer basket, 2021-2024 and 2019-2025 extended panels) | 0 in 749,258 records |
| Incomplete reporter-years | 6, all confirmed against Comtrade as real gaps: Vietnam 2024, 2025; China 2025; Cambodia 2025 (no monthly reports yet); Estonia 2019-12; Philippines 2025-10 |
| Artificial jump where batches meet (Dec 2020→Jan 2021, Dec 2024→Jan 2025) | None; those month-to-month changes are in the normal range |
| Existing rows unchanged by each extension | 100% in every phase (inbound flow, seven features, target) |
| News features for the 18 published countries | 864 of 864 country-months reproduced exactly, in both the 2021–24 and 2019–25 runs |
| Independent recomputation (separate numpy code, own target/lag construction) | All 36 headline numbers match to 3 decimals (consumer basket: 12, 2021-2024 extended panel: 12, 2019-2025 extended panel: 12) |
| Leakage | Features use data up to month T; target uses T+1; scaling and fits use training months only; each rolling block is fit only on months before it |
| Inference clustered by month as well as by country | One claim corrected (section 4) |

Known data caveats
- GDELT has no files for 2025-06-14 to 2025-07-01; June 2025 news volume and tone are interpolated from May and July.
- Romania excluded: GDELT gives Romanian actors no country code.
- The panel target counts imports from the original 18 panel countries only (Taiwan excluded), as in the paper.
- The 41-country news feature for Hong Kong is non-zero in only 65% of months (unchanged from the published panel).

## 2. Panels

| | Original | A | B | D |
|---|---|---|---|---|
| Countries | 18 | 18 | 41 | 41 |
| Baskets | semiconductor | + consumer | both | both |
| Months | 2021-01..2024-12 | same | same | 2019-01..2025-12 |
| Rows per basket | 864 | 864 | 1,968 | 3,444 |
| Valid targets per basket | 643 | 643 | 1,471 | 2,925 |
| Folder | `processed/` | `processed_v2/` | `processed_v3/` | `processed_v4/` |

## 3. Headline results on the non-overlapping target (T1)

Paper protocol (train 2021-12..2022-12, test 2024):

| | Semi, 18 | Consumer, 18 | Semi, 41 | Consumer, 41 | Both, 41 |
|---|---|---|---|---|---|
| Test rows | 187 | 187 | 440 | 440 | 880 |
| Persistence | −1.909 | −1.743 | −1.738 | −1.690 | −1.718 |
| Exogenous only | −0.260 | −0.007 | −0.037 | −0.010 | −0.003 |
| All seven features | −0.250 | 0.093 | 0.051 | 0.188 | 0.120 |
| Trade proxies only | 0.135 | 0.037 | 0.091 | 0.190 | 0.117 |
| Own lags AR(1,12) | 0.215 | 0.200 | 0.161 | 0.168 | 0.166 |

Seven years, ten half-year test blocks (2021H1–2025H2), refit before each block:

| | Semiconductor | Consumer | Both baskets |
|---|---|---|---|
| Test rows | 2,369 | 2,369 | 4,738 |
| Persistence | −1.790 (0/10 blocks positive) | −1.662 (0/10) | −1.753 (0/10) |
| Exogenous only | −0.005 (2/10) | −0.033 (2/10) | −0.012 (2/10) |
| All seven features | 0.116 (9/10) | 0.068 (7/10) | 0.108 (9/10) |
| Trade proxies only | 0.113 (9/10) | 0.121 (8/10) | 0.121 (9/10) |
| Own lags AR(1,12) | 0.172 (8/10) | 0.193 (10/10) | 0.173 (10/10) |
| AR(1,12) + exogenous | 0.166 (8/10) | 0.156 (9/10) | 0.157 (9/10) |

## 4. Significance, with both clusterings (seven-year evaluation, T1)

| Contrast (both baskets) | Difference | p, country-clustered | p, month-clustered | 95% CI (months resampled) |
|---|---|---|---|---|
| Own lags minus all seven | +0.065 | <0.001 | 0.006 | [+0.022, +0.116] |
| All seven minus trade only | −0.013 | <0.001 | 0.249 | [−0.038, +0.007] |
| AR(1,12)+exogenous minus AR(1,12) | −0.016 | <0.001 | 0.083 | [−0.036, +0.001] |
| …news only | −0.001 | 0.004 | 0.067 | [−0.003, +0.000] |
| …weather only | −0.001 | 0.269 | 0.312 | [−0.003, +0.001] |
| …GSCPI only | −0.013 | <0.001 | 0.126 | [−0.032, +0.002] |

**Correction to an earlier statement.** "Adding the exogenous channels measurably hurts (p < 0.001)" was
country-clustered. GSCPI is one global series, so month clustering is the fair test: p = 0.083. The
supported claim is: *the exogenous channels add nothing; with 95% confidence any benefit is at most about
+0.001 R²*. The only significant harm under month clustering is GSCPI on consumer goods (−0.035, p = 0.033).

Claims that hold under both clusterings: own lags beat the seven-feature model (both baskets, consumer);
for semiconductors alone that contrast is not significant (p = 0.32 / 0.075).

The consumer basket's GSCPI hint (+0.031 for consumer goods, 18 countries) did not survive: +0.005 with 41 countries
(p = 0.19) and −0.035 over seven years.

## 5. The overlapping target (T0) in the wider panel

| Panel | Basket | Persistence | All seven | Own lag AR(1) | Note |
|---|---|---|---|---|---|
| 18 countries, test 2024 | semiconductor | 0.648 | 0.338 | 0.558 | paper |
| 18 countries, test 2024 | consumer | 0.497 | 0.198 | 0.389 | |
| 41 countries, test 2024 | semiconductor | 0.257 | −2.975 | 0.047 | Myanmar training values up to 8.2 |
| 41 countries, test 2024, excl. Myanmar | semiconductor | 0.276 | −0.066 | 0.358 | |
| 41 countries, test 2024 | consumer | 0.249 | 0.075 | 0.278 | |
| 41 countries, 10 blocks | semiconductor | 0.268 | −4.302 | 0.253 | one Cambodian month has T0 = 216 |
| 41 countries, 10 blocks | consumer | 0.380 | 0.137 | 0.431 | |

T0 excess kurtosis: 6.4 (original panel) → 122 (41 countries) → 1,589 (41 countries, seven years) for
semiconductors. T0 is unusable for small importers; T1 should be the primary target in the extended results.

## 6. Graph models

| | GCN | TGN | TGN, no memory | Ridge (7 feat.) | Own lags |
|---|---|---|---|---|---|
| Semi, 18 countries, T0 (paper) | 0.328 | 0.308 | 0.314 | 0.338 | 0.563 |
| Semi, 18 countries, T1 | −0.089 | 0.055 | −0.023 | −0.250 | 0.215 |
| Consumer, 18 countries, T0 | 0.376 | 0.332 | 0.320 | 0.198 | 0.526 |
| Consumer, 18 countries, T1 | 0.110 | 0.149 | 0.120 | 0.093 | 0.200 |
| Semi, 41 countries, T0 | 0.044 | 0.134 | 0.132 | −2.975 | 0.041 |
| Semi, 41 countries, T1 | 0.066 | 0.066 | 0.054 | 0.051 | 0.161 |

Five-seed ensembles, the paper's trainer unchanged. On the 41-country T1 panel own lags beat each graph
model (wild-cluster p = 0.005, 0.008, 0.002). Classification on the 41-country semiconductor panel
(91 positives of 446): TGN 0.446, TGN-no-memory 0.444, GCN 0.423, logistic 0.401.

Added 2026-10-07 (runs by `dataset/run_graph_extended.sh`, summary by `dataset/graph_extended_summary.py`,
output `extended_graph_summary.csv`):

41 countries, 2021-24, test 2024, T1 (five seeds). "Both" pools the two baskets' test rows; models are fitted per basket.

| | GCN | TGN | TGN, no memory | Ridge (7 feat.) | Own lags | p own lags vs GCN / TGN / TGN-nm (country; month) |
|---|---|---|---|---|---|---|
| Semiconductor (440 rows) | 0.066 | 0.066 | 0.054 | 0.051 | 0.161 | 0.005 / 0.008 / 0.002; 0.036 / 0.043 / 0.037 |
| Consumer (440 rows) | 0.196 | 0.230 | 0.215 | 0.188 | 0.168 | 0.78 / 0.50 / 0.53; 0.74 / 0.48 / 0.55 |
| Both (880 rows) | 0.114 | 0.126 | 0.113 | 0.101 | 0.164 | 0.005 / 0.034 / 0.006; 0.27 / 0.42 / 0.30 |

Consumer, 41 countries: graph models score highest but are not distinguishable from own lags. TGN vs ridge:
p = 0.013 by country, 0.53 by month.

41 countries, 2019-25, fit 2019-12..2022-12, validation 2023, test 2024-01..2025-11, T1 (five seeds):

| | GCN | TGN | TGN, no memory | Ridge (7 feat.) | Trade only | Own lags | Persistence |
|---|---|---|---|---|---|---|---|
| Semiconductor (894 rows) | 0.136 | 0.115 | 0.115 | 0.133 | 0.140 | 0.191 | -1.779 |
| Consumer (894 rows) | 0.020 | 0.109 | 0.122 | 0.037 | 0.117 | 0.174 | -1.586 |
| Both (1,788 rows) | 0.094 | 0.113 | 0.118 | 0.098 | 0.132 | 0.185 | -1.707 |

p own lags vs GCN / TGN / TGN-nm, country; month: semiconductor 0.150 / 0.131 / 0.053; 0.009 / 0.006 / 0.008.
Consumer <0.001 / 0.015 / 0.125; 0.045 / 0.295 / 0.449. Both <0.001 / <0.001 / <0.001; 0.008 / 0.017 / 0.022.
No graph model differs from the ridge under month clustering (smallest p = 0.052).

41 countries, 2021-24, test 2024, overlapping target T0 (five seeds):

| | GCN | TGN | TGN, no memory | Ridge (7 feat.) | Own lags AR(1,12) | Persistence |
|---|---|---|---|---|---|---|
| Semiconductor (446 rows) | 0.044 | 0.134 | 0.132 | -2.975 | 0.041 | 0.257 |
| Consumer (446 rows) | 0.166 | 0.124 | 0.095 | 0.075 | 0.345 | 0.249 |

Consumer: own lags beat each graph model (p < 0.001 by country; 0.016 / 0.003 / 0.001 by month).
Classification at tau = 0.20, consumer (26 positives of 446), PR-AUC: logistic 0.520, GCN 0.459, TGN-no-memory 0.457, TGN 0.413.

Not run: the ten rolling blocks with graph models, and T0 on the seven-year panel (T0 is unusable there, section 5).

## 6b. Other metrics and fixed settings (added 2026-10-07)

Script `dataset/extended_metrics.py`; outputs `extended_2019_2025/other_metrics.csv`, `other_metrics_tests.csv`,
`sensitivity_alpha_lags.csv`. Both baskets, seven-year panel.

| Model | Ten blocks: R2 / RMSE / MAE / direction % | Test 2024-25: R2 / RMSE / MAE / direction % |
|---|---|---|
| Persistence | -1.753 / 0.367 / 0.236 / 37.1 | -1.707 / 0.363 / 0.246 / 37.5 |
| Exogenous only | -0.012 / 0.223 / 0.144 / 50.7 | not evaluated |
| All seven features | 0.108 / 0.209 / 0.136 / 60.0 | 0.098 / 0.210 / 0.144 / 57.8 |
| Trade proxies only | 0.121 / 0.207 / 0.134 / 61.7 | 0.132 / 0.206 / 0.138 / 61.8 |
| Own lags AR(1,12) | 0.173 / 0.201 / 0.129 / 64.9 | 0.185 / 0.199 / 0.132 / 64.7 |
| AR(1,12) + exogenous | 0.157 / 0.203 / 0.131 / 63.4 | not evaluated |
| GCN / TGN / TGN-no-memory | not evaluated | 0.094, 0.113, 0.118 / 0.210, 0.208, 0.207 / 0.144, 0.140, 0.141 / 56.7, 59.8, 59.7 |

Absolute-error tests (country; month): own lags vs seven features <0.001; <0.001. Own lags vs GCN / TGN / TGN-nm
<0.001 each; 0.003 / 0.006 / 0.005. AR+exogenous vs AR: MAE +0.0025, <0.001; 0.051.

Ridge strength 0.01 to 1000 and lag sets: between alpha 0.01 and 10 no R2 moves by more than 0.006 (either basket);
ordering unchanged up to alpha 100. At alpha 1000 own-lag models are over-shrunk, and on consumer goods own lag 1 (0.084)
and lags 1-3 (0.096) fall below the seven-feature model (0.099). Lags 1,12 vs 1-12: 0.173 vs 0.174 (both),
0.193 vs 0.216 (consumer), 0.172 vs 0.147 (semiconductor). Exogenous only never reaches 0.001.

Graph hyperparameter sensitivity (finished 2026-10-07, 42 runs, all exit 0): `dataset/run_graph_tuning.sh`
(seven variants, three seeds, both baskets), summary `dataset/graph_tuning_summary.py` ->
`extended_2019_2025/graph_tune_{variants,summary}.csv`. Seven-year split, test 2024-01..2025-11, T1, three-seed ensembles.

| Variant | Semi GCN / TGN / TGN-nm | Consumer GCN / TGN / TGN-nm |
|---|---|---|
| Paper setting | 0.138 / 0.115 / 0.115 | 0.035 / 0.108 / 0.120 |
| Learning rate 0.001 | 0.130 / 0.124 / 0.135 | 0.080 / 0.120 / 0.108 |
| Learning rate 0.01 | 0.128 / 0.130 / 0.107 | 0.108 / 0.091 / 0.071 |
| Weight decay 0.001 | 0.135 / 0.136 / 0.131 | 0.048 / 0.100 / 0.118 |
| Half width | 0.131 / 0.122 / 0.103 | 0.107 / -0.055 / 0.103 |
| Double width | 0.117 / 0.114 / 0.126 | 0.118 / 0.049 / 0.114 |
| Squared-error loss | 0.128 / 0.120 / 0.144 | 0.027 / 0.069 / 0.118 |
| 100 epochs, patience 20 | identical to the paper setting (best epoch is within the first few) | same |
| Own lags AR(1,12) | 0.191 | 0.174 |

Both baskets (own lags 0.185): variant chosen on 2023 validation 0.114 / 0.122 / 0.099 (p country <0.001 each;
month 0.013 / 0.019 / 0.015). Variant chosen on test (upper bound, not a valid estimate) 0.131 / 0.130 / 0.136
(p country <=0.002; month 0.033 / 0.036 / 0.077). Paper: `tab:graphtune`.

## 7. Where each number comes from

| Numbers | Script | Output |
|---|---|---|
| Consumer-basket tables | `dataset/basket_analyses.py`, `feature_attribution.py` | `results/benchmark/consumer/` |
| 2021-2024 extended-panel tables | same scripts with `LAMDA_PANEL=processed_v3 LAMDA_OUT=extended_2021_2024`; `extended_power.py` | `results/benchmark/extended_2021_2024/` |
| 2019-2025 extended-panel tables | `dataset/extended_rolling_blocks.py` | `results/benchmark/extended_2019_2025/` |
| Month-clustered tests | (audit) | `extended_2019_2025/exogenous_gain_two_clusterings.csv`, `extended_2019_2025/main_contrasts_two_clusterings.csv` |
| Graph models | `dataset/graph_basket_runs.py`, `graph_basket_summary.py` | `consumer/`, `extended_2021_2024/semiconductor/` |

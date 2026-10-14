# Leakage audit

Task: `classification` on target column `derived_label_tau_0.20`.

## Temporal construction

* Prediction horizon: `1` month(s). The target for month `T` is derived
  from inbound Comtrade flow at `T+1` and a rolling baseline that ends
  at `T+1-1`.
* The model inputs are the 7 fused features of month `T` only:
  `inventory_days_proxy`, `trade_delay_proxy`, `news_vol_7d`, `neg_tone_frac_3d`, `strike_flag_7d`, `weather_anomaly_7d`, `global_risk`.
* `inbound_flow_usd`, `future_inbound_flow_usd`, `baseline_inbound_flow_usd`,
  `contraction`, and every `label_tau_*` column are targets or target
  ingredients. None of them is a model input.
* GCN edges come from the same month's snapshot only.
* TGN events are replayed once, in chronological order, from
  `2021-12` to `2024-11`. Memory is reset before the first
  training month and then carried forward across partition boundaries; it is
  never reset before a validation or test month, and never sees a later month
  before an earlier one.

## Partitions

* Train      : `2021-12` .. `2022-12` (13 months)
* Validation : `2023-01` .. `2023-12` (12 months)
* Test       : `2024-01` .. `2024-11` (11 months)
* Partitions are disjoint and strictly forward-chained. No shuffling, no
  cross-validation folds that mix time, no resampling.
* Months excluded for having no observable target: `2021-01`, `2021-02`, `2021-03`, `2021-04`, `2021-05`, `2021-06`, `2021-07`, `2021-08`, `2021-09`, `2021-10`, `2021-11`, `2024-12`.
* Split notes:
* 12 month(s) have no observable target at this horizon and are excluded from every partition: ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2024-12']
* train: dropped 11 requested month(s) that are absent or have no observable target: ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11']
* test: dropped 1 requested month(s) that are absent or have no observable target: ['2024-12']

## Fitted on training months only

* Feature standardisation (mean and scale).
* Edge log1p scaling for `trade_value_usd` and `flow_volume`.
* The class weight used by the weighted BCE loss.
* The logistic-regression, ridge, and constant baselines.
* Model selection uses the validation partition only. The test partition is
  scored exactly once, with the selected weights.

## Target validity

* Country-month rows: 864
* Valid targets: 643
* Invalid targets: 221
  (future value unobserved: 18,
  baseline unavailable: 198,
  baseline not positive: 6)
* Invalid rows are excluded from training and evaluation. They are never
  converted into negative examples, and no label is synthesised, oversampled,
  or rebalanced.
* All 18 country nodes stay in every graph snapshot so the topology
  is unchanged; only the supervised rows are restricted.

## What this audit does and does not establish

It establishes that the temporal construction and the fitting boundaries are
correct. It does not establish that the resulting sample size or class balance
supports a conclusive performance claim; read `metrics/comparison.csv`
together with `diagnostics/class_balance.json` before quoting any number.

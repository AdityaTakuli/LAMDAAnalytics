# Run artifacts — classification

Generated 2026-09-01T07:24:56.574705+00:00 on `localhost.localdomain` using device
`cpu` in 79.939s.

| Path | Contents |
| --- | --- |
| `run_summary.json` | Everything about this run: options, environment, split, per-model metrics. Start here. |
| `metrics/comparison.csv` | One row per (model, split). The headline column is `average_precision (PR-AUC)`. |
| `metrics/<model>.json` | Full metric block per model, including `note` for undefined metrics. |
| `predictions/<model>.csv` | Per country-month scores: `model, split, month, node_id, target, score`. |
| `checkpoints/<model>.pt` | Weights plus the kwargs, feature order, normaliser, and split needed to reload them. |
| `diagnostics/leakage_audit.md` | What was fitted on what, and why no future information reaches a feature. |
| `diagnostics/target_summary.json` | Target validity, contraction distribution, and class balance. |
| `diagnostics/label_cross_check.json` | Re-derived labels compared against the labels stored in the fused table. |
| `diagnostics/data_validation.json` | Row counts, grid completeness, per-feature ranges, orphan edges. |
| `diagnostics/split.json` | The exact months in each partition, and any month that was dropped. |
| `plots/` | Target distribution, loss curves, model comparison, prediction distributions. |
| `config_used.yaml` | The configuration this run actually saw. |
| `training.log` | The full console log. |

## Reading the numbers honestly

* A metric printed as `null` with a `note` is undefined for that partition, not zero.
* `n` is the number of country-months with an **observable** target. Rows whose
  future value or baseline is missing are excluded, never counted as negatives.
* Test metrics come from a single scoring pass with the validation-selected
  weights. They are not a tuning target.

## Warnings raised by this run

* 12 month(s) have no observable target at this horizon and are excluded from every partition: ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2024-12']
* train: dropped 11 requested month(s) that are absent or have no observable target: ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11']
* test: dropped 1 requested month(s) that are absent or have no observable target: ['2024-12']

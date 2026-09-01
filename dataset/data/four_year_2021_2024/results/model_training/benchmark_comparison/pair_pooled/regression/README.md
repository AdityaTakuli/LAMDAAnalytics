# Pooled bilateral pair-month training run

Supervision unit: directed importer ← exporter links pooled across the panel.
Task: `regression`; headline metric: `rmse`.

## Layout

| Path | Description |
| --- | --- |
| `run_summary.json` | Options, split, metrics, environment |
| `metrics/comparison.csv` | One row per (model, split) |
| `predictions/<model>.csv` | Per pair-month scores |
| `diagnostics/` | Class balance, split, target summary |
| `plots/` | Loss curves, model comparison, target distribution |

Archived country-month runs live under `../country_month/`.

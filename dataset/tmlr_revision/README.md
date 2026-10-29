# TMLR revision analyses

Answers to the reviewer report on the TMLR manuscript (`Research Paper/paper.tex`).
Everything is in this folder; deleting it removes the revision analyses. Nothing
under `results/` or `figures/` is overwritten. Graph models are not retrained:
their saved five-seed predictions in `results/phase07/` are re-scored.

Run from the repository root: `dataset/tmlr_revision/run.sh` (about 15 minutes,
most of it bootstrap loops; `force` reruns, `status` lists outputs).

| Script | Reviewer point | Outputs (`out/`) |
|---|---|---|
| `missing_and_timing.py` | 1, 3, 6 | `reproduce_check.csv` (recomputes 49 canonical R² values, max diff 5e-7), `t0_corrected_numbers.csv` / `t0_corrected_wide.csv` (all T0 numbers with unpublished months treated as missing), `t0_tests.csv`, `per_origin_persistence.csv`, `rolling_origin_models_corrected.csv`, `missing_inventory.csv`, `timing_proxy_check.csv`, `exogenous_strict_timing.csv`, figures |
| `missing_and_timing.py --extras` | 6 | `t0_origin_mean.csv`, `figures/graph_seeds_corrected.pdf`, `figures/rank_inversion_T4_T1.pdf` |
| `dependence_tests.py` | 4 | `dependence_tests.csv`: wild bootstrap by country and by month, two-way cluster-robust t, DM on month means, Holm |
| `selection_checks.py` | 7 | `selection_loco.csv`, `selection_focal.csv`, `selection_focal_members.csv` |

The correction: Comtrade had published no 2024 month for Vietnam when the panel
was downloaded (2026-08-31); the fusion stored those months as zero imports, which
made six 2024 test rows look like total collapses on T0 (value −1). T1, T4 and the
H_k sweep are undefined on those rows and do not change.

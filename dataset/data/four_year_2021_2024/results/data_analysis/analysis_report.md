# 2021–2024 data analysis

Overall training readiness: **READY FOR CONTINUOUS-TARGET PILOT; BINARY CLASSIFICATION BLOCKED**

the training split has only 3 positive tau=0.35 targets; use continuous contraction regression instead of a binary benchmark.

## Gate summary

- Acquisition gate: **PASS** (all Comtrade requests returned successfully).
- Feature gate: **PASS** (four-year processed artifacts and complete 2024 weather coverage required).
- Label gate: **FAIL** (both classes in each split and at least 10 training positives required).
- Continuous-target gate: **PASS** (643 valid non-constant contraction targets support regression).

## Acquisition findings

- Comtrade requests: 960 successful of 960.
- Comtrade records: 31305.
- Reporter-partner-month cells with records: 15845 of 19200 (82.5%); unobserved cells are not failed API requests.
- Successful Comtrade responses with no records: 108.
- GDELT daily export coverage: 1459 of 1461; missing dates: 20221110, 20230323.
- Processed weather profile: 20 locations; missing configured locations: none.

## Target and split findings

- Country-month rows: 960.
- Valid one-month-ahead targets: 643.
- Country-months with zero inbound value: 108.
- Tau 0.35 counts: {"positive": 15, "negative": 628, "prevalence": 0.02332814930015552}.
- train_2021_2022: 234 valid targets, 3 positive, 231 negative.
- validation_2023: 216 valid targets, 5 positive, 211 negative.
- test_2024: 193 valid targets, 7 positive, 186 negative.

## Required next step

The binary classification gate remains blocked by class scarcity. Use the processed profile for a continuous contraction-regression pilot; do not train from raw JSON, ZIP, or Excel files.

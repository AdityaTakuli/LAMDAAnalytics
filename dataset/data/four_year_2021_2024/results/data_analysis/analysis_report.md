# 2021–2024 data analysis

Overall training readiness: **READY FOR TRAINING REVIEW**

Source coverage, target classes, and processed artifacts passed this profile.

## Gate summary

- Acquisition gate: **PASS** (all Comtrade requests returned successfully).
- Feature gate: **PASS** (four-year processed artifacts and complete 2024 weather coverage required).
- Label gate: **PASS** (both classes in each split and at least 10 training positives at tau=0.20).
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
- Classification tau: 0.20 (severe contraction; tau=0.35 has only 15 positives overall).
- Tau 0.20 counts: {"positive": 65, "negative": 578, "prevalence": 0.10108864696734059}.
- train_2021_2022: 234 valid targets, 10 positive, 224 negative.
- validation_2023: 216 valid targets, 36 positive, 180 negative.
- test_2024: 193 valid targets, 19 positive, 174 negative.

## Required next step

Both regression and binary classification benchmarks are supported. Train with `python train_models.py --config config.yaml --task classification`.

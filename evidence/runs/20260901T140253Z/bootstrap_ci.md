# Bootstrap 95% CIs (test 2024, country-month)

Resamples: 5000, seed: 42

## Regression ($R^2$)
| Model | Point | 95% CI | n |
| --- | ---: | --- | ---: |
| ridge_regression | 0.349 | [0.059, 0.502] | 193 |
| tgn | 0.344 | [0.061, 0.493] | 193 |
| gcn | 0.326 | [0.171, 0.409] | 193 |

## Classification (PR-AUC, $\tau=0.20$)
| Model | Point | 95% CI | n |
| --- | ---: | --- | ---: |
| tgn_no_memory | 0.365 | [0.150, 0.560] | 193 |
| logistic_regression | 0.356 | [0.134, 0.561] | 193 |
| tgn | 0.347 | [0.136, 0.546] | 193 |
| gcn | 0.347 | [0.133, 0.549] | 193 |

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
| tgn_no_memory | 0.419 | [0.224, 0.615] | 193 |
| logistic_regression | 0.413 | [0.209, 0.619] | 193 |
| tgn | 0.401 | [0.206, 0.600] | 193 |
| gcn | 0.401 | [0.204, 0.602] | 193 |

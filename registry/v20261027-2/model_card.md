# Model card

Backend: logistic. Evaluated on 4407 walk-forward predictions.

## Feature importance
| feature | log loss increase when shuffled |
|---|---|
| elo_diff | 0.02486 |
| dc_exp_away | 0.01623 |
| dc_exp_home | 0.01396 |
| form_shots_diff_home | 0.00766 |
| form_shots_diff_away | 0.00751 |
| dc_p_away | 0.00485 |
| dc_p_draw | 0.00183 |
| dc_p_home | 0.00135 |
| form_goals_for_home | 1e-05 |
| form_goals_for_away | -0.00011 |
| form_goals_against_home | -0.00021 |
| form_goals_against_away | -0.00023 |
| rest_days_away | -0.00052 |
| rest_days_home | -0.00151 |

## Calibration
| outcome | predicted range | matches | mean predicted | observed |
|---|---|---|---|---|
| home win | 0.0-0.2 | 380 | 0.148 | 0.142 |
| home win | 0.2-0.4 | 1488 | 0.312 | 0.295 |
| home win | 0.4-0.6 | 1644 | 0.49 | 0.47 |
| home win | 0.6-0.8 | 772 | 0.68 | 0.683 |
| home win | 0.8-1.0 | 123 | 0.849 | 0.846 |
| draw | 0.0-0.2 | 574 | 0.156 | 0.169 |
| draw | 0.2-0.4 | 3832 | 0.268 | 0.264 |
| draw | 0.4-0.6 | 1 | 0.407 | 0.0 |
| away win | 0.0-0.2 | 1365 | 0.13 | 0.137 |
| away win | 0.2-0.4 | 1919 | 0.291 | 0.304 |
| away win | 0.4-0.6 | 848 | 0.486 | 0.528 |
| away win | 0.6-0.8 | 256 | 0.667 | 0.652 |
| away win | 0.8-1.0 | 19 | 0.841 | 0.737 |

## Performance by league
| league | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| D1 | 764.0 | 0.9881 | 0.2016 | 0.531 |
| E0 | 952.0 | 1.0024 | 0.2039 | 0.516 |
| F1 | 773.0 | 0.9871 | 0.2058 | 0.542 |
| I1 | 949.0 | 0.984 | 0.1956 | 0.525 |
| SP1 | 969.0 | 0.9733 | 0.1973 | 0.534 |

## Performance by team-strength tier
| tier | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| high | 1469.0 | 0.9261 | 0.1879 | 0.587 |
| low | 1469.0 | 1.0399 | 0.2113 | 0.475 |
| mid | 1469.0 | 0.9946 | 0.2026 | 0.525 |

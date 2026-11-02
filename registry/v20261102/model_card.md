# Model card

Backend: logistic. Evaluated on 4411 walk-forward predictions.

## Feature importance
| feature | log loss increase when shuffled |
|---|---|
| elo_diff | 0.024 |
| dc_exp_away | 0.01422 |
| dc_exp_home | 0.01405 |
| form_shots_diff_away | 0.00913 |
| form_shots_diff_home | 0.00799 |
| dc_p_away | 0.00423 |
| dc_p_home | 0.00161 |
| dc_p_draw | 0.00112 |
| form_goals_for_away | 0.00012 |
| form_goals_for_home | 0.0 |
| form_goals_against_home | -2e-05 |
| rest_days_away | -0.00028 |
| form_goals_against_away | -0.00048 |
| rest_days_home | -0.00187 |

## Calibration
| outcome | predicted range | matches | mean predicted | observed |
|---|---|---|---|---|
| home win | 0.0-0.2 | 399 | 0.146 | 0.133 |
| home win | 0.2-0.4 | 1484 | 0.312 | 0.303 |
| home win | 0.4-0.6 | 1622 | 0.491 | 0.469 |
| home win | 0.6-0.8 | 780 | 0.68 | 0.681 |
| home win | 0.8-1.0 | 126 | 0.849 | 0.849 |
| draw | 0.0-0.2 | 584 | 0.157 | 0.168 |
| draw | 0.2-0.4 | 3826 | 0.267 | 0.264 |
| draw | 0.4-0.6 | 1 | 0.406 | 0.0 |
| away win | 0.0-0.2 | 1371 | 0.129 | 0.141 |
| away win | 0.2-0.4 | 1893 | 0.292 | 0.3 |
| away win | 0.4-0.6 | 856 | 0.487 | 0.525 |
| away win | 0.6-0.8 | 270 | 0.668 | 0.652 |
| away win | 0.8-1.0 | 21 | 0.84 | 0.762 |

## Performance by league
| league | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| D1 | 764.0 | 0.9933 | 0.2028 | 0.526 |
| E0 | 950.0 | 1.0036 | 0.2046 | 0.518 |
| F1 | 772.0 | 0.985 | 0.2056 | 0.545 |
| I1 | 957.0 | 0.9808 | 0.195 | 0.531 |
| SP1 | 968.0 | 0.9728 | 0.1973 | 0.536 |

## Performance by team-strength tier
| tier | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| high | 1470.0 | 0.9231 | 0.1879 | 0.592 |
| low | 1471.0 | 1.0414 | 0.2113 | 0.473 |
| mid | 1470.0 | 0.9961 | 0.2031 | 0.528 |

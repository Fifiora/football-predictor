# Model card

Backend: logistic. Evaluated on 4430 walk-forward predictions.

## Feature importance
| feature | log loss increase when shuffled |
|---|---|
| elo_diff | 0.02542 |
| dc_exp_away | 0.01454 |
| dc_exp_home | 0.01385 |
| form_shots_diff_home | 0.00876 |
| form_shots_diff_away | 0.00639 |
| dc_p_away | 0.00492 |
| dc_p_draw | 0.00229 |
| dc_p_home | 0.00138 |
| form_goals_for_home | 5e-05 |
| form_goals_against_home | 5e-05 |
| form_goals_for_away | 3e-05 |
| form_goals_against_away | 2e-05 |
| rest_days_away | -0.00039 |
| rest_days_home | -0.00109 |

## Calibration
| outcome | predicted range | matches | mean predicted | observed |
|---|---|---|---|---|
| home win | 0.0-0.2 | 404 | 0.146 | 0.134 |
| home win | 0.2-0.4 | 1505 | 0.311 | 0.302 |
| home win | 0.4-0.6 | 1617 | 0.49 | 0.474 |
| home win | 0.6-0.8 | 783 | 0.68 | 0.691 |
| home win | 0.8-1.0 | 121 | 0.85 | 0.835 |
| draw | 0.0-0.2 | 566 | 0.157 | 0.166 |
| draw | 0.2-0.4 | 3863 | 0.267 | 0.262 |
| draw | 0.4-0.6 | 1 | 0.408 | 0.0 |
| away win | 0.0-0.2 | 1377 | 0.129 | 0.14 |
| away win | 0.2-0.4 | 1890 | 0.292 | 0.299 |
| away win | 0.4-0.6 | 871 | 0.487 | 0.525 |
| away win | 0.6-0.8 | 272 | 0.668 | 0.643 |
| away win | 0.8-1.0 | 20 | 0.839 | 0.75 |

## Performance by league
| league | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| D1 | 765.0 | 0.9863 | 0.201 | 0.535 |
| E0 | 956.0 | 1.0036 | 0.2053 | 0.52 |
| F1 | 774.0 | 0.987 | 0.2061 | 0.544 |
| I1 | 965.0 | 0.9792 | 0.1945 | 0.53 |
| SP1 | 970.0 | 0.972 | 0.1965 | 0.535 |

## Performance by team-strength tier
| tier | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| high | 1477.0 | 0.9219 | 0.1874 | 0.591 |
| low | 1477.0 | 1.0378 | 0.2108 | 0.479 |
| mid | 1476.0 | 0.9968 | 0.2032 | 0.526 |

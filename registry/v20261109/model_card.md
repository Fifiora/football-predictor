# Model card

Backend: logistic. Evaluated on 4425 walk-forward predictions.

## Feature importance
| feature | log loss increase when shuffled |
|---|---|
| elo_diff | 0.02716 |
| dc_exp_home | 0.01668 |
| dc_exp_away | 0.01563 |
| form_shots_diff_home | 0.00941 |
| form_shots_diff_away | 0.00627 |
| dc_p_away | 0.00573 |
| dc_p_draw | 0.00185 |
| dc_p_home | 0.00181 |
| form_goals_against_away | 0.00017 |
| form_goals_for_home | 3e-05 |
| form_goals_against_home | -0.00014 |
| form_goals_for_away | -0.00024 |
| rest_days_away | -0.00037 |
| rest_days_home | -0.00133 |

## Calibration
| outcome | predicted range | matches | mean predicted | observed |
|---|---|---|---|---|
| home win | 0.0-0.2 | 400 | 0.145 | 0.135 |
| home win | 0.2-0.4 | 1490 | 0.311 | 0.303 |
| home win | 0.4-0.6 | 1619 | 0.491 | 0.466 |
| home win | 0.6-0.8 | 793 | 0.68 | 0.685 |
| home win | 0.8-1.0 | 123 | 0.849 | 0.846 |
| draw | 0.0-0.2 | 577 | 0.158 | 0.168 |
| draw | 0.2-0.4 | 3847 | 0.266 | 0.264 |
| draw | 0.4-0.6 | 1 | 0.405 | 0.0 |
| away win | 0.0-0.2 | 1374 | 0.128 | 0.139 |
| away win | 0.2-0.4 | 1885 | 0.292 | 0.3 |
| away win | 0.4-0.6 | 872 | 0.487 | 0.526 |
| away win | 0.6-0.8 | 274 | 0.668 | 0.646 |
| away win | 0.8-1.0 | 20 | 0.839 | 0.75 |

## Performance by league
| league | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| D1 | 765.0 | 0.9906 | 0.202 | 0.529 |
| E0 | 957.0 | 1.0023 | 0.2045 | 0.52 |
| F1 | 773.0 | 0.9852 | 0.2061 | 0.546 |
| I1 | 961.0 | 0.9823 | 0.1956 | 0.528 |
| SP1 | 969.0 | 0.9711 | 0.196 | 0.538 |

## Performance by team-strength tier
| tier | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| high | 1475.0 | 0.9224 | 0.188 | 0.592 |
| low | 1475.0 | 1.0427 | 0.211 | 0.473 |
| mid | 1475.0 | 0.9932 | 0.2027 | 0.531 |

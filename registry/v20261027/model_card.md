# Model card

Backend: lightgbm. Evaluated on 929 walk-forward predictions.

## Feature importance
| feature | log loss increase when shuffled |
|---|---|
| form_shots_diff_home | 0.00998 |
| dc_p_home | 0.00975 |
| dc_exp_home | 0.00641 |
| dc_p_away | 0.00604 |
| elo_diff | 0.00535 |
| form_goals_against_away | 0.0011 |
| form_goals_for_away | 0.00088 |
| dc_exp_away | 0.00077 |
| rest_days_away | 0.00026 |
| form_shots_diff_away | -0.0001 |
| rest_days_home | -0.00109 |
| dc_p_draw | -0.00149 |
| form_goals_against_home | -0.00306 |
| form_goals_for_home | -0.00437 |

## Calibration
| outcome | predicted range | matches | mean predicted | observed |
|---|---|---|---|---|
| home win | 0.0-0.2 | 41 | 0.151 | 0.171 |
| home win | 0.2-0.4 | 227 | 0.323 | 0.273 |
| home win | 0.4-0.6 | 467 | 0.485 | 0.407 |
| home win | 0.6-0.8 | 186 | 0.674 | 0.656 |
| home win | 0.8-1.0 | 8 | 0.818 | 1.0 |
| draw | 0.0-0.2 | 208 | 0.178 | 0.221 |
| draw | 0.2-0.4 | 721 | 0.245 | 0.268 |
| away win | 0.0-0.2 | 238 | 0.137 | 0.134 |
| away win | 0.2-0.4 | 509 | 0.295 | 0.344 |
| away win | 0.4-0.6 | 141 | 0.48 | 0.504 |
| away win | 0.6-0.8 | 41 | 0.666 | 0.561 |

## Performance by league
| league | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| E0 | 929.0 | 1.029 | 0.2114 | 0.488 |

## Performance by team-strength tier
| tier | matches | log loss | RPS | top-pick accuracy |
|---|---|---|---|---|
| high | 310.0 | 0.9803 | 0.1967 | 0.542 |
| low | 310.0 | 1.0398 | 0.2128 | 0.474 |
| mid | 309.0 | 1.0671 | 0.2246 | 0.447 |

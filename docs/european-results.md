# Champions League model results

Generated 2026-10-27. Predictions are for the 90-minute result. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE. Models learn from every European competition and are scored on Champions League matches only.

## Development seasons

Everything before 2025-26; tested on the last 3 of them. Publish gate: model log loss 0.970 vs base rate 1.036 and pooled Elo 0.956: does not beat both baselines.

| Model | Matches | Correct picks | Log loss | RPS | ECE |
|---|---|---|---|---|---|
| Base rate | 679 | 47.4% | 1.036 | 0.238 | 0.024 |
| Pooled Elo only | 679 | 54.5% | 0.956 | 0.209 | 0.044 |
| Model | 679 | 54.8% | 0.970 | 0.212 | 0.043 |

Starting-rating offsets fitted on development seasons: E0 +0, SP1 +75, I1 +25, D1 +50, F1 +0, OTHER -75.

## Held-out season

Not evaluated: the model did not pass the publish gate on the development seasons, so 2025-26 stays untouched.

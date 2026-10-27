# Champions League model results

Generated 2026-10-27. Predictions are for the 90-minute result. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE. Models learn from every European competition and are scored on Champions League matches only.

## Development seasons

Everything before 2025-26; tested on the last 3 of them. Publish gate: model log loss 0.904 vs base rate 1.026 and pooled Elo 0.904: passes (beats the base rate, matches pooled Elo).

| Model | Matches | Correct picks | Log loss | RPS | ECE |
|---|---|---|---|---|---|
| Base rate | 439 | 49.7% | 1.026 | 0.235 | 0.025 |
| Pooled Elo only | 439 | 57.6% | 0.904 | 0.192 | 0.044 |
| Model | 439 | 57.6% | 0.904 | 0.192 | 0.044 |

Starting-rating offsets fitted on development seasons: E0 +0, SP1 +25, I1 -25, D1 +0, F1 -75, OTHER -50.

## Held-out season

2025-26 was evaluated once, after the model passed the publish gate. It is not used for tuning.

| Model | Matches | Correct picks | Log loss | RPS | ECE |
|---|---|---|---|---|---|
| Base rate | 189 | 49.7% | 1.024 | 0.235 | 0.022 |
| Pooled Elo only | 189 | 56.1% | 0.958 | 0.210 | 0.054 |
| Model | 189 | 56.1% | 0.958 | 0.210 | 0.054 |

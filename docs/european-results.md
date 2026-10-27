# Champions League model results

Generated 2026-10-27. Predictions are for the 90-minute result. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE. Models learn from every European competition and are scored on Champions League matches only.

## Development seasons

Everything before 2025-26; tested on the last 3 of them. Publish gate: model log loss 0.947 vs base rate 1.036 and pooled Elo 0.947: passes (beats the base rate, matches pooled Elo).

| Model | Matches | Correct picks | Log loss | RPS | ECE |
|---|---|---|---|---|---|
| Base rate | 679 | 47.4% | 1.036 | 0.238 | 0.024 |
| Pooled Elo only | 679 | 56.3% | 0.947 | 0.205 | 0.047 |
| Model | 679 | 56.3% | 0.947 | 0.205 | 0.047 |

Starting-rating offsets fitted on development seasons: E0 +0, SP1 +50, I1 +0, D1 +50, F1 -25, OTHER -75.

## Held-out season

2025-26 was evaluated once, after the model passed the publish gate. It is not used for tuning.

| Model | Matches | Correct picks | Log loss | RPS | ECE |
|---|---|---|---|---|---|
| Base rate | 189 | 49.7% | 1.025 | 0.235 | 0.023 |
| Pooled Elo only | 189 | 52.9% | 0.972 | 0.214 | 0.066 |
| Model | 189 | 52.9% | 0.972 | 0.214 | 0.066 |

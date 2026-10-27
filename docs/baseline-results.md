> **Status: synthetic smoke run only. These are not real football results.**
> The sandbox that built this project cannot reach football-data.co.uk, so the numbers below come
> from a simulated two-league, two-season world (6 teams per league) that exists to prove the
> benchmark code and the hold-out rule work. They say nothing about real-world accuracy.
> To produce the real table, run `python -m footpred.pipeline.cli download` then
> `python -m footpred.pipeline.cli benchmark` (or trigger the retrain workflow) and commit the output.
> The most recent full season is held out and is only looked at if the hybrid wins on the development seasons.

# Baseline versus hybrid results

Generated 2025-10-27. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.

## Development seasons

Everything before 2025-26. All model comparison happens here.

| Model | Matches | Log loss | RPS | ECE | E0 | SP1 |
|---|---|---|---|---|---|---|
| Base rate | 300 | 1.060 | 0.236 | 0.002 | 1.075 | 1.044 |
| Elo only | 300 | 1.052 | 0.234 | 0.027 | 1.081 | 1.023 |
| Dixon-Coles only | 300 | 1.039 | 0.229 | 0.066 | 1.112 | 0.966 |
| Hybrid | 300 | 1.144 | 0.242 | 0.089 | 1.184 | 1.105 |

## Held-out season

Not evaluated: the hybrid did not beat both Elo-only and Dixon-Coles-only on the development seasons, so 2025-26 stays untouched.

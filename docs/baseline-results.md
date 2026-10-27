# Baseline versus hybrid results

Generated 2026-10-27. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.

## Development seasons

Everything before 2025-26. All model comparison happens here.

| Model | Matches | Log loss | RPS | ECE | E0 |
|---|---|---|---|---|---|
| Base rate | 380 | 1.081 | 0.235 | 0.027 | 1.081 |
| Elo only | 380 | 0.998 | 0.206 | 0.041 | 0.998 |
| Dixon-Coles only | 380 | 0.990 | 0.205 | 0.034 | 0.990 |
| Hybrid | 380 | 1.022 | 0.210 | 0.048 | 1.022 |

## Held-out season

Not evaluated: the hybrid did not beat both Elo-only and Dixon-Coles-only on the development seasons, so 2025-26 stays untouched.

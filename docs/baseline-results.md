# Baseline versus hybrid results

Generated 2026-10-27. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.

## Development seasons

Everything before 2025-26; tested on the last 3 of them. All model comparison happens here. Best hybrid variant on these seasons: hybrid_logistic.

| Model | Matches | Log loss | RPS | ECE | D1 | E0 | F1 | I1 | SP1 |
|---|---|---|---|---|---|---|---|---|---|
| Base rate | 5330 | 1.073 | 0.230 | 0.005 | 1.074 | 1.063 | 1.073 | 1.087 | 1.067 |
| Elo only | 5330 | 0.991 | 0.202 | 0.011 | 1.006 | 0.973 | 1.001 | 0.991 | 0.986 |
| Dixon-Coles only | 5330 | 0.987 | 0.201 | 0.013 | 1.002 | 0.974 | 0.999 | 0.985 | 0.978 |
| Hybrid | 5330 | 0.998 | 0.200 | 0.014 | 1.041 | 0.972 | 1.010 | 0.991 | 0.985 |
| Hybrid (logistic) | 5330 | 0.997 | 0.200 | 0.010 | 1.043 | 0.974 | 1.003 | 0.996 | 0.978 |
| Hybrid (regularized trees) | 5330 | 1.000 | 0.200 | 0.009 | 1.020 | 0.988 | 1.015 | 1.000 | 0.984 |

## Held-out season

Not evaluated: the hybrid did not beat both Elo-only and Dixon-Coles-only on the development seasons, so 2025-26 stays untouched.

# Baseline versus hybrid results

Generated 2026-10-27. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.

## Development seasons

Everything before 2025-26; tested on the last 3 of them. All model comparison happens here. Best hybrid variant on these seasons: hybrid_logistic.

| Model | Matches | Log loss | RPS | ECE | E0 |
|---|---|---|---|---|---|
| Base rate | 1140 | 1.062 | 0.233 | 0.003 | 1.062 |
| Elo only | 1140 | 0.972 | 0.201 | 0.018 | 0.972 |
| Dixon-Coles only | 1140 | 0.974 | 0.202 | 0.019 | 0.974 |
| Hybrid | 1140 | 0.982 | 0.202 | 0.022 | 0.982 |
| Hybrid (logistic) | 1140 | 0.968 | 0.199 | 0.025 | 0.968 |
| Hybrid (regularized trees) | 1140 | 0.973 | 0.200 | 0.023 | 0.973 |

## Held-out season

2025-26 was evaluated once, after the hybrid beat Elo-only and Dixon-Coles-only on the development seasons. Only the chosen variant (hybrid_logistic) was evaluated, shown as Hybrid. It is not used for tuning; this is the first release candidate.

| Model | Matches | Log loss | RPS | ECE | E0 |
|---|---|---|---|---|---|
| Base rate | 460 | 1.087 | 0.228 | 0.028 | 1.087 |
| Elo only | 460 | 1.033 | 0.210 | 0.033 | 1.033 |
| Dixon-Coles only | 460 | 1.039 | 0.210 | 0.051 | 1.039 |
| Hybrid | 460 | 1.040 | 0.213 | 0.043 | 1.040 |

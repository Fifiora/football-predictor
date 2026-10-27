# Baseline versus hybrid results

Generated 2026-10-27. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.

## Development seasons

Everything before 2025-26; tested on the last 3 of them. All model comparison happens here. Best hybrid variant on these seasons: hybrid_logistic.

| Model | Matches | Log loss | RPS | ECE | D1 | E0 | F1 | I1 | SP1 |
|---|---|---|---|---|---|---|---|---|---|
| Base rate | 5330 | 1.073 | 0.230 | 0.005 | 1.074 | 1.063 | 1.073 | 1.087 | 1.067 |
| Elo only | 5330 | 0.991 | 0.202 | 0.011 | 1.006 | 0.973 | 1.001 | 0.991 | 0.986 |
| Dixon-Coles only | 5330 | 0.987 | 0.201 | 0.013 | 1.002 | 0.974 | 0.999 | 0.985 | 0.978 |
| Blend (Elo + Dixon-Coles) | 5330 | 0.983 | 0.200 | 0.010 | 0.998 | 0.967 | 0.995 | 0.983 | 0.979 |
| Hybrid | 5330 | 0.985 | 0.200 | 0.014 | 0.992 | 0.970 | 1.004 | 0.986 | 0.976 |
| Hybrid (blend backend, calibrated) | 5330 | 0.984 | 0.200 | 0.010 | 1.000 | 0.968 | 0.997 | 0.982 | 0.979 |
| Hybrid (logistic, core inputs only) | 5330 | 0.984 | 0.200 | 0.009 | 0.999 | 0.968 | 0.997 | 0.982 | 0.978 |
| Hybrid (logistic) | 5330 | 0.981 | 0.199 | 0.009 | 0.992 | 0.961 | 0.998 | 0.982 | 0.977 |
| Hybrid (regularized trees) | 5330 | 0.983 | 0.199 | 0.015 | 0.991 | 0.969 | 1.000 | 0.985 | 0.974 |

## Held-out season

2025-26 was evaluated once, after the hybrid beat Elo-only and Dixon-Coles-only on the development seasons. Only the chosen variant (hybrid_logistic) was evaluated, shown as Hybrid. It is not used for tuning; this is the first release candidate.

| Model | Matches | Log loss | RPS | ECE | D1 | E0 | F1 | I1 | SP1 |
|---|---|---|---|---|---|---|---|---|---|
| Base rate | 2146 | 1.071 | 0.229 | 0.001 | 1.065 | 1.084 | 1.064 | 1.083 | 1.056 |
| Elo only | 2146 | 1.001 | 0.205 | 0.012 | 0.978 | 1.031 | 1.005 | 0.992 | 0.994 |
| Dixon-Coles only | 2146 | 0.995 | 0.203 | 0.021 | 0.989 | 1.039 | 0.997 | 0.978 | 0.975 |
| Blend (Elo + Dixon-Coles) | 2146 | 0.992 | 0.202 | 0.018 | 0.977 | 1.024 | 0.997 | 0.981 | 0.979 |
| Hybrid | 2146 | 0.992 | 0.202 | 0.019 | 0.975 | 1.044 | 0.993 | 0.969 | 0.979 |

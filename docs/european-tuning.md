# Champions League tuning (development seasons only)

Seasons before 2025-26; walk-forward from 2022-09-07. Lower log loss is better.

| k | home adv | regress | variant | matches | correct picks | log loss |
|---|---|---|---|---|---|---|
| 25 | 50 | 0.15 | elo_diff only C=10 uncalibrated | 679 | 56.3% | 0.9476 |
| 25 | 50 | 0.15 | pooled Elo baseline | 679 | 56.3% | 0.9477 |
| 25 | 80 | 0.15 | elo_diff only C=10 uncalibrated | 679 | 56.1% | 0.9477 |
| 25 | 80 | 0.15 | pooled Elo baseline | 679 | 56.1% | 0.9477 |
| 25 | 50 | 0.15 | all3 C=1 uncalibrated | 679 | 55.7% | 0.9517 |
| 25 | 80 | 0.15 | all3 C=1 uncalibrated | 679 | 55.7% | 0.9517 |
| 25 | 80 | 0.15 | all3 C=10 uncalibrated | 679 | 55.7% | 0.9518 |
| 25 | 50 | 0.15 | all3 C=10 uncalibrated | 679 | 55.7% | 0.9518 |
| 15 | 50 | 0.15 | elo_diff only C=10 uncalibrated | 679 | 55.5% | 0.9523 |
| 15 | 50 | 0.15 | pooled Elo baseline | 679 | 55.5% | 0.9523 |
| 15 | 80 | 0.15 | elo_diff only C=10 uncalibrated | 679 | 55.4% | 0.9524 |
| 15 | 80 | 0.15 | pooled Elo baseline | 679 | 55.4% | 0.9524 |
| 25 | 50 | 0.3 | pooled Elo baseline | 679 | 54.8% | 0.9558 |
| 25 | 50 | 0.3 | elo_diff only C=10 uncalibrated | 679 | 54.8% | 0.9558 |
| 15 | 80 | 0.15 | all3 C=1 uncalibrated | 679 | 55.5% | 0.9565 |
| 15 | 80 | 0.15 | all3 C=10 uncalibrated | 679 | 55.5% | 0.9566 |
| 15 | 50 | 0.15 | all3 C=1 uncalibrated | 679 | 55.5% | 0.9566 |
| 15 | 50 | 0.15 | all3 C=10 uncalibrated | 679 | 55.5% | 0.9567 |
| 25 | 80 | 0.3 | pooled Elo baseline | 679 | 54.6% | 0.9582 |
| 25 | 80 | 0.3 | elo_diff only C=10 uncalibrated | 679 | 54.6% | 0.9582 |
| 25 | 50 | 0.3 | all3 C=1 uncalibrated | 679 | 54.3% | 0.9596 |
| 25 | 50 | 0.3 | all3 C=10 uncalibrated | 679 | 54.5% | 0.9597 |
| 25 | 80 | 0.3 | all3 C=1 uncalibrated | 679 | 54.5% | 0.9619 |
| 25 | 80 | 0.3 | all3 C=10 uncalibrated | 679 | 54.5% | 0.9620 |
| 15 | 50 | 0.3 | pooled Elo baseline | 679 | 54.6% | 0.9634 |
| 15 | 50 | 0.3 | elo_diff only C=10 uncalibrated | 679 | 54.6% | 0.9634 |
| 25 | 50 | 0.15 | all3 C=0.1 calibrated (current) | 679 | 55.5% | 0.9638 |
| 25 | 50 | 0.15 | all3 C=1 calibrated | 679 | 55.5% | 0.9638 |
| 15 | 80 | 0.3 | pooled Elo baseline | 679 | 54.2% | 0.9640 |
| 15 | 80 | 0.3 | elo_diff only C=10 uncalibrated | 679 | 54.2% | 0.9640 |
| 25 | 80 | 0.15 | all3 C=1 calibrated | 679 | 55.2% | 0.9645 |
| 25 | 80 | 0.15 | all3 C=0.1 calibrated (current) | 679 | 55.2% | 0.9646 |
| 15 | 50 | 0.3 | all3 C=1 uncalibrated | 679 | 54.2% | 0.9674 |
| 15 | 50 | 0.3 | all3 C=10 uncalibrated | 679 | 54.2% | 0.9675 |
| 15 | 80 | 0.3 | all3 C=1 uncalibrated | 679 | 54.2% | 0.9680 |
| 15 | 80 | 0.3 | all3 C=10 uncalibrated | 679 | 54.1% | 0.9681 |
| 25 | 50 | 0.3 | all3 C=1 calibrated | 679 | 54.6% | 0.9695 |
| 25 | 50 | 0.3 | all3 C=0.1 calibrated (current) | 679 | 54.6% | 0.9696 |
| 15 | 80 | 0.15 | all3 C=0.1 calibrated (current) | 679 | 55.2% | 0.9698 |
| 15 | 80 | 0.15 | all3 C=1 calibrated | 679 | 55.2% | 0.9700 |
| 15 | 50 | 0.15 | all3 C=0.1 calibrated (current) | 679 | 55.1% | 0.9703 |
| 15 | 50 | 0.15 | all3 C=1 calibrated | 679 | 55.1% | 0.9703 |
| 25 | 80 | 0.3 | all3 C=1 calibrated | 679 | 54.1% | 0.9720 |
| 25 | 80 | 0.3 | all3 C=0.1 calibrated (current) | 679 | 54.1% | 0.9721 |
| 15 | 50 | 0.3 | all3 C=1 calibrated | 679 | 53.9% | 0.9770 |
| 15 | 50 | 0.3 | all3 C=0.1 calibrated (current) | 679 | 53.9% | 0.9770 |
| 15 | 80 | 0.3 | all3 C=1 calibrated | 679 | 54.6% | 0.9778 |
| 15 | 80 | 0.3 | all3 C=0.1 calibrated (current) | 679 | 54.6% | 0.9779 |
| 25 | 50 | 0.15 | base rate | 679 | 47.4% | 1.0360 |
| 15 | 80 | 0.3 | base rate | 679 | 47.4% | 1.0360 |
| 15 | 50 | 0.3 | base rate | 679 | 47.4% | 1.0360 |
| 15 | 80 | 0.15 | base rate | 679 | 47.4% | 1.0360 |
| 15 | 50 | 0.15 | base rate | 679 | 47.4% | 1.0360 |
| 25 | 50 | 0.3 | base rate | 679 | 47.4% | 1.0360 |
| 25 | 80 | 0.15 | base rate | 679 | 47.4% | 1.0360 |
| 25 | 80 | 0.3 | base rate | 679 | 47.4% | 1.0360 |

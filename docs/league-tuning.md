# League tuning (development seasons only)

Seasons before 2025-26; walk-forward from 2022-08-05. Lower log loss is better. The hold-out season is not read here.

## Stage 1: Elo settings (Elo-only logistic)

| k | home adv | regress | margin | correct picks | log loss |
|---|---|---|---|---|---|
| 15 | 70 | 0.15 | True | 53.1% | 0.9893 |
| 15 | 50 | 0.15 | True | 52.9% | 0.9893 |
| 15 | 70 | 0.25 | True | 53.2% | 0.9898 |
| 15 | 50 | 0.25 | True | 53.0% | 0.9898 |
| 20 | 70 | 0.15 | True | 53.2% | 0.9899 |
| 20 | 50 | 0.15 | True | 53.2% | 0.9900 |
| 20 | 70 | 0.15 | False | 53.1% | 0.9901 |
| 20 | 50 | 0.15 | False | 53.1% | 0.9901 |
| 30 | 70 | 0.15 | False | 53.0% | 0.9905 |
| 20 | 70 | 0.25 | False | 53.1% | 0.9905 |
| 20 | 50 | 0.25 | False | 53.2% | 0.9906 |
| 30 | 50 | 0.15 | False | 52.9% | 0.9906 |
| 20 | 70 | 0.25 | True | 53.0% | 0.9907 |
| 15 | 70 | 0.35 | True | 53.2% | 0.9907 |
| 15 | 50 | 0.35 | True | 53.3% | 0.9907 |
| 20 | 50 | 0.25 | True | 52.9% | 0.9908 |
| 15 | 50 | 0.25 | False | 53.2% | 0.9909 |
| 15 | 70 | 0.25 | False | 53.2% | 0.9909 |
| 15 | 50 | 0.15 | False | 53.1% | 0.9909 |
| 15 | 70 | 0.15 | False | 53.2% | 0.9909 |
| 30 | 70 | 0.25 | False | 52.8% | 0.9912 |
| 20 | 70 | 0.35 | False | 53.0% | 0.9913 |
| 15 | 70 | 0.35 | False | 53.1% | 0.9914 |
| 30 | 50 | 0.25 | False | 52.8% | 0.9914 |
| 20 | 50 | 0.35 | False | 53.0% | 0.9914 |
| 15 | 50 | 0.35 | False | 53.1% | 0.9914 |
| 20 | 70 | 0.35 | True | 52.9% | 0.9917 |
| 20 | 50 | 0.35 | True | 52.9% | 0.9918 |
| 30 | 70 | 0.35 | False | 52.8% | 0.9923 |
| 30 | 50 | 0.35 | False | 52.7% | 0.9925 |
| 30 | 70 | 0.15 | True | 52.7% | 0.9926 |
| 30 | 50 | 0.15 | True | 52.7% | 0.9928 |
| 30 | 70 | 0.25 | True | 52.8% | 0.9935 |
| 30 | 50 | 0.25 | True | 52.7% | 0.9937 |
| 30 | 70 | 0.35 | True | 52.8% | 0.9946 |
| 30 | 50 | 0.35 | True | 52.9% | 0.9948 |

## Stage 2: goal-model decay (Dixon-Coles only)

| xi | correct picks | log loss |
|---|---|---|
| 0.0018 | 52.7% | 0.9865 |
| 0.0025 | 52.9% | 0.9866 |
| 0.0013 | 52.6% | 0.9874 |
| 0.0035 | 52.7% | 0.9880 |
| 0.0008 | 52.2% | 0.9897 |

## Stage 3: hybrid inputs (Elo EloParams(k=15.0, home_adv=70.0, season_regress=0.15, initial=1500.0, margin=True), xi 0.0018)

| inputs | C | correct picks | log loss |
|---|---|---|---|
| current inputs | 0.03 | 53.1% | 0.9813 |
| current inputs | 0.1 | 53.2% | 0.9814 |
| current inputs | 0.3 | 53.2% | 0.9815 |
| current + new form inputs | 0.03 | 53.1% | 0.9816 |
| current + new form inputs | 0.1 | 53.1% | 0.9818 |
| current + new form inputs | 0.3 | 53.0% | 0.9818 |
| without shot form | 0.03 | 53.2% | 0.9841 |
| without shot form | 0.1 | 53.2% | 0.9843 |
| without shot form | 0.3 | 53.2% | 0.9843 |
| without shot form + new form inputs | 0.03 | 53.1% | 0.9844 |
| without shot form + new form inputs | 0.1 | 53.1% | 0.9845 |
| without shot form + new form inputs | 0.3 | 53.0% | 0.9846 |

Chosen: Elo k=15, home adv 70, regress 0.15, margin True; xi 0.0018; inputs 'current inputs' at C=0.03.

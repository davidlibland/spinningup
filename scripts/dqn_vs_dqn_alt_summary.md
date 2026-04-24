# DQN (stock) vs dqn_alt.py on CartPole-v1

- Env: CartPole-v1 (solved threshold = 475, max return 500)
- Total env-step budget per run: 500,000
- Seeds per algo: 5 ([1, 2, 3, 4, 5])
- Final-window mean: average episodic return over last 50,000 steps

## Aggregated metrics

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| DQN (stock CleanRL) | 493.7 +/- 5.2 | 500.0 +/- 0.0 | 159,742 | 5/5 | 38.3 |
| DQN (dqn_alt.py) | 220.8 +/- 78.0 | 500.0 +/- 0.0 | 347,492 | 2/2 | 268.1 |

## Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| DQN (stock CleanRL) | 1 | 488.0 | 500.0 | 123,303 | 37.7 |
| DQN (stock CleanRL) | 2 | 490.7 | 500.0 | 182,425 | 40.7 |
| DQN (stock CleanRL) | 3 | 500.0 | 500.0 | 159,742 | 39.2 |
| DQN (stock CleanRL) | 4 | 489.5 | 500.0 | 164,168 | 36.7 |
| DQN (stock CleanRL) | 5 | 500.0 | 500.0 | 147,077 | 37.3 |
| DQN (dqn_alt.py) | 1 | 298.8 | 500.0 | 389,311 | 274.4 |
| DQN (dqn_alt.py) | 2 | 142.8 | 500.0 | 305,674 | 261.7 |

# CartPole-v1 Algorithm Comparison

- Total timesteps per run: 500,000
- Seeds per algorithm: 5 ([1, 2, 3, 4, 5])
- "Solved" threshold: episodic return >= 475
- Final-window mean: average episodic return over last 25,000 steps

## Aggregated metrics

| Algorithm | Final return (mean +/- std) | Deterministic test return (mean +/- std) | Steps to solve (median) | Solved seeds | Wall-clock (s, mean) | Late-training std (mean) |
|---|---|---|---|---|---|---|
| PPO | 484.7 +/- 8.5 | 500.0 +/- 0.0 | 20,404 | 5/5 | 35.5 | 59.2 |
| DQN | 493.6 +/- 5.3 | 500.0 +/- 0.0 | 159,742 | 5/5 | 38.3 | 31.2 |
| C51 | 468.9 +/- 47.4 | 500.0 +/- 0.0 | 173,379 | 5/5 | 104.5 | 38.9 |
| PQN | 496.4 +/- 7.2 | 500.0 +/- 0.0 | 142,020 | 5/5 | 26.6 | 9.9 |

## Per-run detail

| Algorithm | Seed | Final return | Test return | First solved step | Duration (s) |
|---|---|---|---|---|---|
| PPO | 1 | 470.0 | 500.0 | 14,572 | 33.8 |
| PPO | 2 | 495.8 | 500.0 | 26,500 | 34.9 |
| PPO | 3 | 489.6 | 500.0 | 15,024 | 35.6 |
| PPO | 4 | 484.5 | 500.0 | 20,404 | 36.9 |
| PPO | 5 | 483.4 | 500.0 | 21,884 | 36.6 |
| DQN | 1 | 489.1 | 500.0 | 123,303 | 37.7 |
| DQN | 2 | 488.3 | 500.0 | 182,425 | 40.7 |
| DQN | 3 | 500.0 | 500.0 | 159,742 | 39.2 |
| DQN | 4 | 490.5 | 500.0 | 164,168 | 36.7 |
| DQN | 5 | 500.0 | 500.0 | 147,077 | 37.3 |
| C51 | 1 | 500.0 | 500.0 | 162,530 | 103.7 |
| C51 | 2 | 500.0 | 500.0 | 175,370 | 105.8 |
| C51 | 3 | 377.7 | 500.0 | 182,142 | 104.1 |
| C51 | 4 | 500.0 | 500.0 | 153,310 | 101.3 |
| C51 | 5 | 466.9 | 500.0 | 173,379 | 107.5 |
| PQN | 1 | 482.1 | 500.0 | 286,928 | 26.3 |
| PQN | 2 | 500.0 | 500.0 | 118,716 | 25.9 |
| PQN | 3 | 500.0 | 500.0 | 125,740 | 26.9 |
| PQN | 4 | 500.0 | 500.0 | 142,020 | 26.4 |
| PQN | 5 | 500.0 | 500.0 | 165,576 | 27.7 |

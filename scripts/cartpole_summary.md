# CartPole-v1 Algorithm Comparison

- Total timesteps per run: 200,000
- Seeds per algorithm: 5 ([1, 2, 3, 4, 5])
- "Solved" threshold: episodic return >= 475
- Final-window mean: average episodic return over last 10,000 steps

## Aggregated metrics

| Algorithm | Final return (mean +/- std) | Sample efficiency: steps to solve (median) | Solved seeds | Wall-clock (s, mean) | Within-run late-training std (mean) |
|---|---|---|---|---|---|
| PPO | 312.6 +/- 107.3 | 21,108 | 5/5 | 14.2 | 78.8 |
| DQN | 441.7 +/- 95.0 | 59,556 | 4/5 | 14.8 | 30.6 |
| C51 | 431.5 +/- 32.1 | 118,324 | 5/5 | 40.0 | 85.6 |
| PQN | 193.2 +/- 138.1 | 72,704 | 4/5 | 11.2 | 37.2 |

## Per-run detail

| Algorithm | Seed | Final return | First solved step | Duration (s) |
|---|---|---|---|---|
| PPO | 1 | 264.9 | 20,308 | 14.4 |
| PPO | 2 | 169.3 | 14,788 | 14.1 |
| PPO | 3 | 367.9 | 29,736 | 14.5 |
| PPO | 4 | 274.2 | 21,108 | 14.4 |
| PPO | 5 | 486.4 | 27,812 | 13.8 |
| DQN | 1 | 252.1 | never | 15.5 |
| DQN | 2 | 496.9 | 63,093 | 15.0 |
| DQN | 3 | 489.6 | 60,744 | 14.6 |
| DQN | 4 | 480.3 | 58,369 | 14.5 |
| DQN | 5 | 489.8 | 51,980 | 14.6 |
| C51 | 1 | 372.2 | 118,324 | 39.5 |
| C51 | 2 | 457.7 | 85,303 | 38.9 |
| C51 | 3 | 429.4 | 126,172 | 39.5 |
| C51 | 4 | 461.5 | 82,282 | 39.1 |
| C51 | 5 | 436.7 | 119,027 | 42.8 |
| PQN | 1 | 469.0 | 79,636 | 11.4 |
| PQN | 2 | 129.0 | 63,416 | 11.2 |
| PQN | 3 | 118.4 | 65,772 | 10.7 |
| PQN | 4 | 116.5 | never | 11.3 |
| PQN | 5 | 133.0 | 102,508 | 11.3 |

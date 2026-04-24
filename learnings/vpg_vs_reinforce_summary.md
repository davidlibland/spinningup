# VPG vs REINFORCE on CartPole-v1

- Env: CartPole-v1 (solved threshold = 475, max return 500)
- Total env-step budget per run: 500,000
- Seeds per algo: 5 ([1, 2, 3, 4, 5])
- Final-window mean: average episodic return over last 50,000 steps

## Aggregated metrics

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| REINFORCE | 448.8 +/- 18.1 | 500.0 +/- 0.0 | 229,532 | 5/5 | 17.9 |
| VPG | 443.4 +/- 21.4 | 500.0 +/- 0.0 | 230,888 | 5/5 | 22.2 |

## Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| REINFORCE | 1 | 438.3 | 500.0 | 200,824 | 17.5 |
| REINFORCE | 2 | 420.1 | 500.0 | 250,124 | 17.7 |
| REINFORCE | 3 | 453.2 | 500.0 | 225,456 | 18.6 |
| REINFORCE | 4 | 472.6 | 500.0 | 229,532 | 17.7 |
| REINFORCE | 5 | 459.8 | 500.0 | 256,184 | 17.9 |
| VPG | 1 | 442.7 | 500.0 | 221,064 | 39.9 |
| VPG | 2 | 406.0 | 500.0 | 253,468 | 18.7 |
| VPG | 3 | 455.9 | 500.0 | 216,540 | 17.2 |
| VPG | 4 | 441.8 | 500.0 | 230,888 | 17.6 |
| VPG | 5 | 470.5 | 500.0 | 263,800 | 17.4 |

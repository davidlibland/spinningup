# PPO vs REINFORCE vs REINFORCE+baseline

- Total env-step budget per run: 500,000
- Seeds per (env, algo) cell: 5 ([1, 2, 3, 4, 5])
- Final-window mean: average episodic return over last 50,000 steps


## CartPole-v1 (solved = 475)

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| PPO | 490.0 +/- 5.3 | 500.0 +/- 0.0 | 20,404 | 5/5 | 33.9 |
| REINFORCE | 448.8 +/- 18.1 | 500.0 +/- 0.0 | 229,532 | 5/5 | 17.9 |
| REINFORCE + baseline | 491.8 +/- 6.0 | 500.0 +/- 0.0 | 67,856 | 5/5 | 23.4 |

### Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| PPO | 1 | 480.4 | 500.0 | 14,572 | 34.4 |
| PPO | 2 | 494.8 | 500.0 | 26,500 | 32.1 |
| PPO | 3 | 494.8 | 500.0 | 15,024 | 35.6 |
| PPO | 4 | 491.4 | 500.0 | 20,404 | 33.7 |
| PPO | 5 | 488.5 | 500.0 | 21,884 | 33.8 |
| REINFORCE | 1 | 438.3 | 500.0 | 200,824 | 17.5 |
| REINFORCE | 2 | 420.1 | 500.0 | 250,124 | 17.7 |
| REINFORCE | 3 | 453.2 | 500.0 | 225,456 | 18.6 |
| REINFORCE | 4 | 472.6 | 500.0 | 229,532 | 17.7 |
| REINFORCE | 5 | 459.8 | 500.0 | 256,184 | 17.9 |
| REINFORCE + baseline | 1 | 496.1 | 500.0 | 70,628 | 23.4 |
| REINFORCE + baseline | 2 | 495.2 | 500.0 | 71,168 | 23.3 |
| REINFORCE + baseline | 3 | 498.5 | 500.0 | 67,856 | 23.7 |
| REINFORCE + baseline | 4 | 483.4 | 500.0 | 66,920 | 23.0 |
| REINFORCE + baseline | 5 | 485.7 | 500.0 | 67,016 | 23.3 |

## Acrobot-v1 (solved = -100)

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| PPO | -84.6 +/- 1.0 | -74.8 +/- 7.8 | 78,140 | 5/5 | 46.7 |
| REINFORCE | -192.3 +/- 116.7 | -245.0 +/- 208.3 | 362,548 | 3/5 | 30.0 |
| REINFORCE + baseline | -86.9 +/- 2.0 | -80.0 +/- 4.8 | 65,116 | 5/5 | 37.2 |

### Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| PPO | 1 | -86.2 | -79.0 | 153,328 | 46.3 |
| PPO | 2 | -85.0 | -85.0 | 72,236 | 46.8 |
| PPO | 3 | -83.3 | -69.0 | 93,244 | 46.6 |
| PPO | 4 | -84.3 | -63.0 | 51,196 | 47.0 |
| PPO | 5 | -84.3 | -78.0 | 78,140 | 47.0 |
| REINFORCE | 1 | -108.4 | -79.0 | 246,376 | 30.1 |
| REINFORCE | 2 | -115.1 | -500.0 | 362,548 | 30.2 |
| REINFORCE | 3 | -182.5 | -66.0 | never | 29.7 |
| REINFORCE | 4 | -419.8 | -500.0 | never | 30.1 |
| REINFORCE | 5 | -135.8 | -80.0 | 374,292 | 29.8 |
| REINFORCE + baseline | 1 | -90.5 | -79.0 | 98,464 | 36.6 |
| REINFORCE + baseline | 2 | -84.5 | -81.0 | 65,116 | 36.8 |
| REINFORCE + baseline | 3 | -86.0 | -88.0 | 91,100 | 37.6 |
| REINFORCE + baseline | 4 | -86.8 | -73.0 | 53,576 | 37.5 |
| REINFORCE + baseline | 5 | -86.5 | -79.0 | 54,992 | 37.4 |

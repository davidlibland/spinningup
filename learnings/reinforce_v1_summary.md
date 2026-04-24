# PPO vs REINFORCE vs REINFORCE+baseline

- Total env-step budget per run: 500,000
- Seeds per (env, algo) cell: 5 ([1, 2, 3, 4, 5])
- Final-window mean: average episodic return over last 50,000 steps


## CartPole-v1 (solved = 475)

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| PPO | 490.0 +/- 5.3 | 500.0 +/- 0.0 | 20,404 | 5/5 | 33.9 |
| REINFORCE | 35.5 +/- 0.7 | 150.6 +/- 91.7 | never | 0/5 | 18.6 |
| REINFORCE + baseline | 51.8 +/- 0.9 | 130.6 +/- 41.3 | never | 0/5 | 21.6 |

### Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| PPO | 1 | 480.4 | 500.0 | 14,572 | 34.4 |
| PPO | 2 | 494.8 | 500.0 | 26,500 | 32.1 |
| PPO | 3 | 494.8 | 500.0 | 15,024 | 35.6 |
| PPO | 4 | 491.4 | 500.0 | 20,404 | 33.7 |
| PPO | 5 | 488.5 | 500.0 | 21,884 | 33.8 |
| REINFORCE | 1 | 36.3 | 78.0 | never | 18.7 |
| REINFORCE | 2 | 35.7 | 323.0 | never | 18.3 |
| REINFORCE | 3 | 35.9 | 165.0 | never | 18.5 |
| REINFORCE | 4 | 34.3 | 81.0 | never | 18.8 |
| REINFORCE | 5 | 35.1 | 106.0 | never | 18.8 |
| REINFORCE + baseline | 1 | 52.2 | 81.0 | never | 21.8 |
| REINFORCE + baseline | 2 | 53.1 | 148.0 | never | 21.6 |
| REINFORCE + baseline | 3 | 51.8 | 144.0 | never | 21.7 |
| REINFORCE + baseline | 4 | 51.8 | 88.0 | never | 21.7 |
| REINFORCE + baseline | 5 | 50.3 | 192.0 | never | 21.4 |

## Acrobot-v1 (solved = -100)

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| PPO | -84.6 +/- 1.0 | -74.8 +/- 7.8 | 78,140 | 5/5 | 46.7 |
| REINFORCE | -498.2 +/- 0.8 | -500.0 +/- 0.0 | never | 0/5 | 29.9 |
| REINFORCE + baseline | -499.3 +/- 0.9 | -423.6 +/- 152.8 | never | 0/5 | 33.6 |

### Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| PPO | 1 | -86.2 | -79.0 | 153,328 | 46.3 |
| PPO | 2 | -85.0 | -85.0 | 72,236 | 46.8 |
| PPO | 3 | -83.3 | -69.0 | 93,244 | 46.6 |
| PPO | 4 | -84.3 | -63.0 | 51,196 | 47.0 |
| PPO | 5 | -84.3 | -78.0 | 78,140 | 47.0 |
| REINFORCE | 1 | -497.0 | -500.0 | never | 29.6 |
| REINFORCE | 2 | -498.8 | -500.0 | never | 29.9 |
| REINFORCE | 3 | -499.0 | -500.0 | never | 30.0 |
| REINFORCE | 4 | -497.4 | -500.0 | never | 30.4 |
| REINFORCE | 5 | -498.9 | -500.0 | never | 29.7 |
| REINFORCE + baseline | 1 | -500.0 | -500.0 | never | 33.7 |
| REINFORCE + baseline | 2 | -499.8 | -500.0 | never | 33.7 |
| REINFORCE + baseline | 3 | -499.5 | -118.0 | never | 33.5 |
| REINFORCE + baseline | 4 | -499.5 | -500.0 | never | 33.5 |
| REINFORCE + baseline | 5 | -497.6 | -500.0 | never | 33.5 |

## LunarLander-v3 (solved = 200)

| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |
|---|---|---|---|---|---|
| PPO | 21.2 +/- 18.2 | -45.6 +/- 22.6 | 410,286 | 2/5 | 41.3 |
| REINFORCE | -154.8 +/- 3.6 | -104.0 +/- 95.7 | never | 0/5 | 26.6 |
| REINFORCE + baseline | -141.2 +/- 2.8 | -22.0 +/- 158.8 | never | 0/5 | 30.4 |

### Per-seed detail

| Algo | Seed | Final | Test | First solved | Duration (s) |
|---|---|---|---|---|---|
| PPO | 1 | 9.1 | -35.3 | never | 41.1 |
| PPO | 2 | 22.8 | -90.7 | 352,132 | 41.5 |
| PPO | 3 | 19.4 | -31.0 | never | 41.2 |
| PPO | 4 | 0.7 | -36.9 | never | 41.0 |
| PPO | 5 | 54.1 | -34.0 | 468,440 | 41.9 |
| REINFORCE | 1 | -153.1 | -101.5 | never | 26.6 |
| REINFORCE | 2 | -159.1 | -30.0 | never | 26.6 |
| REINFORCE | 3 | -149.5 | -190.0 | never | 26.7 |
| REINFORCE | 4 | -158.5 | -227.3 | never | 26.8 |
| REINFORCE | 5 | -153.5 | 29.1 | never | 26.4 |
| REINFORCE + baseline | 1 | -140.7 | 119.6 | never | 30.5 |
| REINFORCE + baseline | 2 | -141.5 | -68.6 | never | 30.2 |
| REINFORCE + baseline | 3 | -136.9 | 156.0 | never | 30.3 |
| REINFORCE + baseline | 4 | -141.5 | -26.5 | never | 30.3 |
| REINFORCE + baseline | 5 | -145.6 | -290.7 | never | 30.5 |

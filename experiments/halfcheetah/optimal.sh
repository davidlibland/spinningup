#!/usr/bin/env bash
# Best HalfCheetah-v4 PPO config from the optuna sweep + 5-seed reeval.
# Winner = trial #33 (reeval mean 1727 vs sweep value 2888).
# Run from this directory.
set -euo pipefail

TOTAL_TIMESTEPS=1000000
SEED="${SEED:-1}"

# --- Hyperparameters that the ablation showed actually matter (load-bearing
# vs the cleanrl ppo_continuous_action.py defaults — flipping any one of
# these back to its default produces a clean, tight-CI regression):
LR=0.0003291092148347622           # ~ cleanrl default (3e-4) — incidental
CLIP_COEF=0.3143701180151749       # vs default 0.2  — load-bearing
ENT_COEF=0.0020314931914659313     # vs default 0.0  — load-bearing
# --no-anneal-lr (vs default True) — load-bearing

# --- Hyperparameters the ablation showed were noise / possibly over-tuned
# by optuna (no clean regression when flipped to default — keep them at the
# winner values for fidelity but they aren't important):
NUM_ENVS=2
NUM_STEPS=1024
NUM_MINIBATCHES=8
UPDATE_EPOCHS=20                   # ablation hint: 10 may be no worse
VF_COEF=0.3478327044080488
GAMMA=0.9913956035800737           # ~ cleanrl default (0.99) — incidental
GAE_LAMBDA=0.8981594396023258

uv run python ../../algos/ppo_continuous_action.py \
    --env-id HalfCheetah-v4 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --seed "${SEED}" \
    --exp-name "optimal_winner_t33" \
    --learning-rate "${LR}" \
    --num-envs "${NUM_ENVS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --update-epochs "${UPDATE_EPOCHS}" \
    --clip-coef "${CLIP_COEF}" \
    --ent-coef "${ENT_COEF}" \
    --vf-coef "${VF_COEF}" \
    --gamma "${GAMMA}" \
    --gae-lambda "${GAE_LAMBDA}" \
    --no-anneal-lr \
    --no-clip-vloss \
    --normalize \
    --eval-episodes 10 \
    --capture-video \
    --capture-test-video

# --- Reeval results (5 seeds × 500k steps, DET_EVAL mean):
#   #33  1727 ± [1388, 2299]  std 640   (this config)
#   #79  1624 ± [1474, 1786]  std 204   (tighter; consider if you want
#                                        reliability over peak — see below)
#   #27  1465 ± [1409, 1531]  std  78
#
# --- Alternative: sweep trial #79 (tighter variance across seeds)
#   --learning-rate 0.00021447356483155928 --num-envs 8 --num-steps 256 \
#   --num-minibatches 64 --update-epochs 17 --clip-coef 0.20792395284331072 \
#   --ent-coef 0.008237989019603029 --vf-coef 0.5554260824414656 \
#   --gamma 0.9473270735505909 --gae-lambda 0.9057323987499132 \
#   --anneal-lr --clip-vloss --normalize
#
# --- Ablation (LOO around #33, 3 seeds × 500k steps, DET_EVAL mean ± 95% CI):
#   A_winner                       2114 [1504, 3076]   baseline
#   flip_clip_coef→0.2             1569 [1538, 1598]   -546   load-bearing
#   flip_ent_coef→0.0              1584 [1344, 1753]   -530   load-bearing
#   flip_anneal_lr→True            1528 [1429, 1586]   -587   load-bearing
#   flip_update_epochs→10          3048 [1614, 3820]   +934   probably over-tuned
#   flip_vf_coef→0.5               2439 [1409, 4249]   +324   noisy, ambiguous
#   flip_clip_vloss→True           2444 [1586, 4026]   +330   noisy, ambiguous
#   flip_rollout→default            2082 [ 976, 4082]   -33    noisy, ambiguous
#   Y_cleanrl_defaults             1417 [1284, 1503]   -697   stock cleanrl
#   Z_sb3_zoo_halfcheetah          1066 [1022, 1109]  -1049   (designed for 1M)

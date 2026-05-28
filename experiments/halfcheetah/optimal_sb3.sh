#!/usr/bin/env bash
# SB3-zoo HalfCheetah-v4 PPO hyperparameters, ported to this repo's
# algos/ppo_continuous_action.py. Source: hyperparams/ppo.yml in
# DLR-RM/rl-baselines3-zoo (HalfCheetah-v4 section, "# Tuned").
# Run from this directory.
set -euo pipefail

TOTAL_TIMESTEPS=1000000   # SB3's published training budget for HalfCheetah-v4
SEED="${SEED:-1}"

# SB3 YAML -> CLI mapping:
#   batch_size: 64      -> num_minibatches = n_envs * n_steps / 64 = 1 * 512 / 64 = 8
#   learning_rate: 2.0633e-05  (constant; no `lin_` prefix in upstream)  -> --no-anneal-lr
#   clip_range: 0.1            -> --clip-coef
#   (no clip_range_vf)         -> --no-clip-vloss
#   normalize: true            -> --normalize
LR=2.0633e-05
NUM_ENVS=1
NUM_STEPS=512
NUM_MINIBATCHES=8
UPDATE_EPOCHS=20
CLIP_COEF=0.1
ENT_COEF=0.000401762
VF_COEF=0.58096
GAMMA=0.98
GAE_LAMBDA=0.92
MAX_GRAD_NORM=0.8

uv run python ../../algos/ppo_continuous_action.py \
    --env-id HalfCheetah-v4 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --seed "${SEED}" \
    --exp-name "optimal_sb3_zoo" \
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
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --no-anneal-lr \
    --no-clip-vloss \
    --normalize \
    --eval-episodes 10 \
    --capture-video \
    --capture-test-video

# Caveat — architecture differences (NOT portable via CLI):
#   SB3 PPO MlpPolicy on HalfCheetah-v4 uses:
#     net_arch = pi=[256,256], vf=[256,256]   (cleanrl: 2×64)
#     activation_fn = nn.ReLU                 (cleanrl: nn.Tanh)
#     log_std_init = -2                       (cleanrl: 0)
#     ortho_init = False                      (cleanrl: True with sqrt(2) gains)
#   These knobs are hard-coded in ../../algos/ppo_continuous_action.py — i.e.
#   this script ports the *algorithmic* hparams but not the policy network.
#   SB3's published HalfCheetah-v4 score (~1976) was achieved with the full
#   stack, so do not expect bitwise replication.

#!/usr/bin/env bash
# Run every discrete-action algorithm in algos/ against LunarLander-v3 with 5 seeds.
# 4 algorithms x 5 seeds = 20 runs.
# Run from the repo root.
set -euo pipefail

# cd "$(dirname "$0")/.."

TOTAL_TIMESTEPS=100000
export TOTAL_TIMESTEPS

parallel --jobs 6 --progress \
    'uv run python "../../algos/{1}.py" \
        --env-id Pendulum-v1 \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --seed {2} \
        --learning-rate {3} \
        --exp-name "ppo-gamma{5}-no-anneal" \
        --no-anneal-lr \
        --num-envs 2 \
        --gamma {5} \
        --num-steps {4} \
        --update-epochs 10 \
        --no-normalize \
        --no-clip_vloss \
        --capture-video \
        --capture-test-video' \
    ::: ppo_continuous_action \
    ::: 1 \
    ::: 0.001 \
    ::: 1024 \
    ::: 0.8 0.9 0.95
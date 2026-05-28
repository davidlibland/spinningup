#!/usr/bin/env bash
# Run every discrete-action algorithm in algos/ against LunarLander-v3 with 5 seeds.
# 4 algorithms x 5 seeds = 20 runs.
# Run from the repo root.
set -euo pipefail

# cd "$(dirname "$0")/.."

TOTAL_TIMESTEPS=1000000
export TOTAL_TIMESTEPS

parallel --jobs 12 --progress \
    'uv run python "../../algos/{1}.py" \
        --env-id HalfCheetah-v4 \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --seed {2} \
        --learning-rate {3} \
        --exp-name "longer_run_{3}_e{4}_s{5}" \
        --no-anneal-lr \
        --num-envs {4} \
        --num-steps {5} \
        --capture-video \
        --capture-test-video' \
    ::: ppo_continuous_action \
    ::: 1 \
    ::: 0.001 0.0003 0.0001 \
    ::: 1 3 9 \
    ::: 128 512 2048
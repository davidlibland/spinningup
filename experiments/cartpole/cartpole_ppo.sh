#!/usr/bin/env bash
# Run every discrete-action algorithm in algos/ against CartPole-v1 with 5 seeds.
# 4 algorithms x 5 seeds = 20 runs.
# Run from the repo root.
set -euo pipefail

# cd "$(dirname "$0")/.."

TOTAL_TIMESTEPS=500000
export TOTAL_TIMESTEPS

parallel --jobs 5 --progress \
    'uv run python "../../algos/{1}.py" \
        --env-id CartPole-v1 \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --seed {2} \
        --learning-rate {3} \
        --capture-video \
        --capture-test-video' \
    ::: ppo \
    ::: 1 2 3 4 5 \
    ::: 0.001 0.0003
#!/usr/bin/env bash
# Run every discrete-action algorithm in algos/ against LunarLander-v3 with 5 seeds.
# 4 algorithms x 5 seeds = 20 runs.
# Run from the repo root.
set -euo pipefail

# cd "$(dirname "$0")/.."

TOTAL_TIMESTEPS=800000
export TOTAL_TIMESTEPS

parallel --jobs 6 --progress \
    'uv run python "../../algos/{1}.py" \
        --env-id LunarLander-v3 \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --seed {2} \
        --learning-rate {3} \
        --exp-name "ppo-lr{3}-nsteps{4}-anneal" \
        --anneal-lr \
        --num-steps {4} \
        --update-epochs 10 \
        --capture-video \
        --capture-test-video' \
    ::: ppo \
    ::: 1 2 3 \
    ::: 0.0003 \
    ::: 256 512 1024
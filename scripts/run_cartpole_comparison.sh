#!/usr/bin/env bash
# Run every discrete-action algorithm in algos/ against CartPole-v1 with 5 seeds.
# 4 algorithms x 5 seeds = 20 runs. Sequential, so wall-clock comparisons are fair.
# Run from the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

ALGOS=(ppo dqn c51 pqn)
SEEDS=(1 2 3 4 5)
TOTAL_TIMESTEPS=500000

run_id=0
total_runs=$(( ${#ALGOS[@]} * ${#SEEDS[@]} ))

for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_id=$(( run_id + 1 ))
        echo ""
        echo "=== [${run_id}/${total_runs}] ${algo} seed=${seed} ==="
        uv run python "algos/${algo}.py" \
            --env-id CartPole-v1 \
            --total-timesteps "${TOTAL_TIMESTEPS}" \
            --seed "${seed}" \
            --capture-video \
            --capture-test-video
        echo "--- [${run_id}/${total_runs}] ${algo} seed=${seed} DONE ---"
    done
done

echo ""
echo "All ${total_runs} runs complete."

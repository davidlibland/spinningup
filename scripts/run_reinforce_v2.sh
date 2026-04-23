#!/usr/bin/env bash
# REINFORCE re-run with tuned defaults.
# 2 envs (CartPole, Acrobot) x 2 algos (reinforce, reinforce_with_baseline) x 5 seeds = 20 runs.
# PPO baselines already copied into runs/ from the prior experiment (same PPO hyperparams).
# All per-algo hyperparameters are now defaults in the scripts themselves — no overrides needed.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=".:${PYTHONPATH:-}"

SEEDS=(1 2 3 4 5)
TOTAL_TIMESTEPS=500000
ALGOS=(reinforce reinforce_with_baseline)
ENVS=(CartPole-v1 Acrobot-v1)

run_id=0
total=$(( ${#ALGOS[@]} * ${#ENVS[@]} * ${#SEEDS[@]} ))

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_id=$(( run_id + 1 ))
            echo ""
            echo "=== [${run_id}/${total}] ${algo} env=${env} seed=${seed} ==="
            uv run python "algos/${algo}.py" \
                --env-id "${env}" \
                --total-timesteps "${TOTAL_TIMESTEPS}" \
                --seed "${seed}" \
                --capture-test-video
            echo "--- [${run_id}/${total}] ${algo} env=${env} seed=${seed} DONE ---"
        done
    done
done

echo ""
echo "All ${total} runs complete."

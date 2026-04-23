#!/usr/bin/env bash
# Apples-to-apples comparison: PPO vs. REINFORCE vs. REINFORCE+baseline
# across three discrete-action classic-control envs. 5 seeds per cell, 500k steps each.
# Equalized env-step budget; per-algo num_steps / update_epochs / num_minibatches vary
# to fit each method's natural operating point.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=".:${PYTHONPATH:-}"

SEEDS=(1 2 3 4 5)
TOTAL_TIMESTEPS=500000

# Each config: "algo env num_steps num_minibatches update_epochs"
# - ppo: published defaults (128 / 4 / 4)
# - reinforce: big rollouts so fewer trajectories get zero-bootstrapped at truncation
# - reinforce_with_baseline: V(s) bootstrap handles truncation, so moderate rollouts
configs=(
    "ppo                     CartPole-v1      128  4 4"
    "reinforce               CartPole-v1     1024  1 1"
    "reinforce_with_baseline CartPole-v1      256  1 1"
    "ppo                     Acrobot-v1       128  4 4"
    "reinforce               Acrobot-v1      1024  1 1"
    "reinforce_with_baseline Acrobot-v1       256  1 1"
    "ppo                     LunarLander-v3   128  4 4"
    "reinforce               LunarLander-v3  1024  1 1"
    "reinforce_with_baseline LunarLander-v3   256  1 1"
)

run_id=0
total=$(( ${#configs[@]} * ${#SEEDS[@]} ))

for config in "${configs[@]}"; do
    read -r algo env num_steps num_mb update_epochs <<< "$config"
    for seed in "${SEEDS[@]}"; do
        run_id=$(( run_id + 1 ))
        echo ""
        echo "=== [${run_id}/${total}] ${algo} env=${env} seed=${seed} num_steps=${num_steps} ==="
        uv run python "algos/${algo}.py" \
            --env-id "${env}" \
            --total-timesteps "${TOTAL_TIMESTEPS}" \
            --num-steps "${num_steps}" \
            --num-minibatches "${num_mb}" \
            --update-epochs "${update_epochs}" \
            --seed "${seed}" \
            --capture-test-video
        echo "--- [${run_id}/${total}] ${algo} env=${env} seed=${seed} DONE ---"
    done
done

echo ""
echo "All ${total} runs complete."

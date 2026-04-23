#!/usr/bin/env bash
# Smoke-test each ported algo for a handful of timesteps.
# Exits on first failure. Run from the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:."

run() {
    local name="$1"
    shift
    echo ""
    echo "=== $name ==="
    "$@"
    echo "--- $name PASSED ---"
}

# Classic control on-policy
run "ppo.py" uv run python algos/ppo.py --total-timesteps 2048 --num-envs 4 --num-steps 128
run "pqn.py" uv run python algos/pqn.py --total-timesteps 2048

# Classic control off-policy (single env, small replay, quick training)
run "dqn.py" uv run python algos/dqn.py --total-timesteps 300 --learning-starts 100 --train-frequency 25 --target-network-frequency 100 --batch-size 32
run "c51.py" uv run python algos/c51.py --total-timesteps 300 --learning-starts 100 --train-frequency 25 --target-network-frequency 100 --batch-size 32

# Continuous control. These need mujoco for HalfCheetah; fall back to Pendulum-v1.
run "ppo_continuous_action.py (Pendulum)" uv run python algos/ppo_continuous_action.py --env-id Pendulum-v1 --total-timesteps 2048 --num-envs 1 --num-steps 2048
run "rpo_continuous_action.py (Pendulum)" uv run python algos/rpo_continuous_action.py --env-id Pendulum-v1 --total-timesteps 2048 --num-envs 1 --num-steps 2048

# Off-policy continuous
run "ddpg_continuous_action.py (Pendulum)" uv run python algos/ddpg_continuous_action.py --env-id Pendulum-v1 --total-timesteps 300 --learning-starts 100 --batch-size 32
run "td3_continuous_action.py (Pendulum)" uv run python algos/td3_continuous_action.py --env-id Pendulum-v1 --total-timesteps 300 --learning-starts 100 --batch-size 32
run "sac_continuous_action.py (Pendulum)" uv run python algos/sac_continuous_action.py --env-id Pendulum-v1 --total-timesteps 300 --learning-starts 100 --batch-size 32

echo ""
echo "All smoke tests passed."

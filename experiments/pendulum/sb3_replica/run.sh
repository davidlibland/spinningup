#!/usr/bin/env bash
# EXACT SB3-zoo Pendulum PPO+gSDE config + SB3 truncation bootstrapping.
set -euo pipefail
cd "$(dirname "$0")/.."
for S in 101 102 103; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python ../../algos/ppo_gSDE.py \
    --env-id Pendulum-v1 --total-timesteps 100000 --seed "$S" \
    --num-envs 4 --num-steps 1024 --num-minibatches 64 --update-epochs 10 \
    --learning-rate 0.001 --no-anneal-lr --gamma 0.9 --gae-lambda 0.95 \
    --clip-coef 0.2 --no-clip-vloss --ent-coef 0.0 --vf-coef 0.5 \
    --max-grad-norm 0.5 --use-sde --sde-sample-freq 4 --sde-log-std-init 0.0 \
    --no-normalize --eval-episodes 20 \
    --no-capture-video --no-capture-test-video \
    --exp-name "sb3replica-s$S" \
    --optuna-report-path "sb3_replica/seed$S.jsonl" \
    > "sb3_replica/seed$S.log" 2>&1 &
done
wait
echo ALL DONE

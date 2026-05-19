#!/usr/bin/env bash
# SB3-zoo Pendulum gSDE recipe, NO obs/reward normalization, 100k, 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
for S in 101 102 103; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python ../../algos/ppo_gSDE.py \
    --env-id Pendulum-v1 --total-timesteps 100000 --seed "$S" \
    --num-envs 4 --num-steps 1024 --num-minibatches 64 --update-epochs 10 \
    --learning-rate 0.001 --gamma 0.9 --gae-lambda 0.95 \
    --ent-coef 0.0 --clip-coef 0.2 --vf-coef 0.5 --no-anneal-lr \
    --use-sde --sde-sample-freq 4 --no-normalize \
    --no-capture-video --no-capture-test-video \
    --exp-name "gsde-nonorm-s$S" \
    --optuna-report-path "gsde_nonorm/seed$S.jsonl" \
    > "gsde_nonorm/seed$S.log" 2>&1 &
done
wait
echo ALL DONE

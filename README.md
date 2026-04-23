# Install
```bash
bashuv sync
uv sync --group dev
```
This creates `.venv/` automatically. Activate with `source .venv/bin/activate` or just prefix commands with `uv run`.

## Grab CleanRL's single-file scripts
CleanRL's design philosophy is that you copy the .py you want and own it — don't import it. So:
```bash
# Just clone for reference and copy what you need
git clone --depth 1 https://github.com/vwxyzjn/cleanrl.git /tmp/cleanrl-ref
mkdir -p algos
cp /tmp/cleanrl-ref/cleanrl/ppo_continuous_action.py algos/
cp /tmp/cleanrl-ref/cleanrl/sac_continuous_action.py algos/
cp /tmp/cleanrl-ref/cleanrl/ddpg_continuous_action.py algos/
cp /tmp/cleanrl-ref/cleanrl/td3_continuous_action.py algos/
cp /tmp/cleanrl-ref/cleanrl/ppo.py algos/  # discrete version
```

## Smoke test:
```bash
uv run python algos/ppo.py --env-id CartPole-v1 --total-timesteps 50000
uv run python algos/ppo_continuous_action.py --env-id Pendulum-v1 --total-timesteps 100000
uv run python algos/sac_continuous_action.py --env-id HalfCheetah-v5 --total-timesteps 100000
```
(and in another terminal)
```bash
uv run tensorboard --logdir runs
```

## To capture video:
```bash
uv run python algos/ppo_continuous_action.py \
    --env-id HalfCheetah-v5 \
    --total-timesteps 1_000_000 \
    --capture-video
```
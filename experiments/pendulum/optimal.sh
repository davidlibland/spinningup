lr=0.001
nsteps=1024
ecoef=0.002
gamma=0.9

uv run python ../../algos/ppo_continuous_action.py \
        --env-id Pendulum-v1 \
        --total-timesteps 100000 \
        --learning-rate $lr \
        --exp-name "sweep_winner_v2" \
        --no-anneal-lr \
        --num-steps $nsteps \
        --update-epochs 10 \
        --num-envs 2 \
        --ent-coef $ecoef \
        --gamma $gamma \
        --no-normalize \
        --no-clip_vloss \
        --capture-video \
        --capture-test-video \
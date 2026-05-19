lr=0.001
nsteps=128
ecoef=0.0005

uv run python ../../algos/ppo.py \
        --env-id LunarLander-v3 \
        --total-timesteps 1000000 \
        --learning-rate $lr \
        --exp-name "sweep_winner" \
        --anneal-lr \
        --num-steps $nsteps \
        --update-epochs 10 \
        --num-envs 16 \
        --vf-coef 1.32 \
        --no-anneal-lr \
        --ent-coef $ecoef \
        --capture-video \
        --capture-test-video \
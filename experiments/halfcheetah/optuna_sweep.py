"""Optuna sweep over continuous-PPO hyperparameters on HalfCheetah-v4.

Mirrors experiments/pendulum/optuna_sweep.py: each trial spawns
`algos/ppo_continuous_action.py` as a subprocess with sampled hparams. The
`--optuna-report-path` JSONL stream (per-iteration training rolling-mean) is
used ONLY by the MedianPruner for early stopping. The trial OBJECTIVE is the
DETERMINISTIC evaluation (`--eval-episodes`, policy mean, no exploration
noise) parsed from the trial log.

Budget on this box (RTX 3090 Ti / 32-core): the existing 1M-step HalfCheetah
runs in cheetah.sh (12 parallel jobs) hit ~690 SPS at num_envs=9, ~234 SPS
at num_envs=3, ~78 SPS at num_envs=1 — training cost amortizes over the
larger rollout batch, so SPS scales near-linearly with num_envs. For a
~1-hour total wall budget on 500k steps/trial, num_envs=1 alone would burn
the whole hour on a single trial, so the sweep excludes num_envs=1 and
defaults to n_jobs=6 (≈5 cores/trial). Expect roughly 3-15 min/trial
depending on num_envs/num_steps, with the pruner killing weak runs early.

Search space tracks both cleanrl defaults (lr 3e-4, num_envs 1, num_steps
2048, num_minibatches 32, anneal_lr True, clip_vloss True, normalize True)
and the SB3-zoo HalfCheetah-v4 values (lr 2.06e-5 linear, batch_size 64,
n_epochs 20, gamma 0.98, gae_lambda 0.92, clip_range 0.1,
ent_coef 4.02e-4).

Run from this directory: `uv run python optuna_sweep.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PPO_SCRIPT = REPO_ROOT / "algos" / "ppo_continuous_action.py"

DEFAULT_STORAGE = f"sqlite:///{HERE / 'optuna_studies' / 'halfcheetah_ppo.db'}"
DEFAULT_STUDY = "halfcheetah_ppo_v1"
POLL_INTERVAL_S = 2.0
TERMINATE_GRACE_S = 30.0
EVAL_EPISODES = 10  # deterministic eval episodes; this is the trial objective

DET_EVAL_RE = re.compile(
    r"DET_EVAL episodes=\d+ mean=(-?[\d.]+) std=(-?[\d.]+) "
    r"min=(-?[\d.]+) max=(-?[\d.]+)"
)


def sample_params(trial: optuna.Trial) -> dict:
    """PPO knobs for HalfCheetah-v4 (ppo_continuous_action.py).

    Ranges span cleanrl defaults and SB3-zoo HalfCheetah values so TPE can
    rediscover either. num_envs=1 is excluded to keep per-trial wall time
    inside the ~1h sweep budget — at num_envs=1 a single trial of 500k
    steps takes ~30+ min on this box. γ and λ are sampled log-uniformly in
    `1-γ` / `1-λ` space for uniform coverage in effective-horizon space.
    max_grad_norm=0.5 and norm_adv=True are held fixed (cleanrl & SB3
    agree).
    """
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 1e-3, 1e-1, log=True)
    one_minus_lambda = trial.suggest_float("one_minus_lambda", 5e-3, 2e-1, log=True)
    gamma = 1.0 - one_minus_gamma
    gae_lambda = 1.0 - one_minus_lambda
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("gae_lambda", gae_lambda)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "num_envs": trial.suggest_categorical("num_envs", [2, 4, 8, 16]),
        "num_steps": trial.suggest_categorical(
            "num_steps", [128, 256, 512, 1024, 2048]
        ),
        "num_minibatches": trial.suggest_categorical(
            "num_minibatches", [4, 8, 16, 32, 64]
        ),
        "update_epochs": trial.suggest_int("update_epochs", 3, 20),
        "clip_coef": trial.suggest_float("clip_coef", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.0),
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "anneal_lr": trial.suggest_categorical("anneal_lr", [True, False]),
        "clip_vloss": trial.suggest_categorical("clip_vloss", [True, False]),
        "normalize": trial.suggest_categorical("normalize", [True, False]),
    }


def build_cmd(params: dict, *, total_timesteps: int, seed: int, exp_name: str,
              report_path: Path) -> list[str]:
    return [
        "uv", "run", "python", str(PPO_SCRIPT),
        "--env-id", "HalfCheetah-v4",
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
        "--num-envs", str(params["num_envs"]),
        "--exp-name", exp_name,
        "--no-capture-video", "--no-capture-test-video",
        "--optuna-report-path", str(report_path),
        "--eval-episodes", str(EVAL_EPISODES),
        "--learning-rate", repr(params["learning_rate"]),
        "--num-steps", str(params["num_steps"]),
        "--num-minibatches", str(params["num_minibatches"]),
        "--update-epochs", str(params["update_epochs"]),
        "--clip-coef", repr(params["clip_coef"]),
        "--ent-coef", repr(params["ent_coef"]),
        "--vf-coef", repr(params["vf_coef"]),
        "--gae-lambda", repr(params["gae_lambda"]),
        "--gamma", repr(params["gamma"]),
        "--anneal-lr" if params["anneal_lr"] else "--no-anneal-lr",
        "--clip-vloss" if params["clip_vloss"] else "--no-clip-vloss",
        "--normalize" if params["normalize"] else "--no-normalize",
    ]


def consume_reports(fh, trial: optuna.Trial) -> bool:
    """Read all currently-available JSONL lines, call trial.report().

    Uses `readline()` so the file can be tailed across multiple polls.
    Returns True if Optuna requests pruning."""
    pruned = False
    while True:
        line = fh.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = int(entry["global_step"])
        value = float(entry["return"])
        trial.report(value, step)
        if trial.should_prune():
            pruned = True
    return pruned


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait()


def read_all_reports(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def make_objective(args):
    reports_dir = HERE / "optuna_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_dir = HERE / "optuna_trial_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        report_path = reports_dir / f"trial_{trial.number:04d}.jsonl"
        if report_path.exists():
            report_path.unlink()
        exp_name = f"optuna-t{trial.number:04d}"
        seed = args.seed_base + trial.number
        cmd = build_cmd(params, total_timesteps=args.total_timesteps,
                        seed=seed, exp_name=exp_name, report_path=report_path)

        print(f"\n[trial {trial.number}] starting: seed={seed}")
        print(f"  params: {params}")
        print(f"  cmd: {shlex.join(cmd)}")
        log_path = log_dir / f"trial_{trial.number:04d}.log"
        log_fh = log_path.open("w")
        env = {**os.environ,
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )

        report_fh = None
        pruned = False
        try:
            while proc.poll() is None:
                if report_fh is None and report_path.exists():
                    report_fh = report_path.open()
                if report_fh is not None:
                    if consume_reports(report_fh, trial):
                        pruned = True
                        break
                time.sleep(POLL_INTERVAL_S)
            if report_fh is None and report_path.exists():
                report_fh = report_path.open()
            if report_fh is not None:
                consume_reports(report_fh, trial)
        finally:
            if report_fh is not None:
                report_fh.close()
            if pruned:
                terminate(proc)
            else:
                proc.wait()
            log_fh.close()

        if pruned:
            raise optuna.TrialPruned()
        if proc.returncode != 0:
            print(f"[trial {trial.number}] subprocess failed (rc={proc.returncode}); "
                  f"see {log_path}")
            raise optuna.TrialPruned()

        # Objective = the DETERMINISTIC evaluation (policy mean, no
        # exploration noise) printed by ppo_continuous_action.py at the end.
        # The JSONL stream above is used ONLY for the pruner's intermediate
        # values; it is not the trial score.
        m = DET_EVAL_RE.search(log_path.read_text())
        if m is None:
            print(f"[trial {trial.number}] no DET_EVAL line emitted; "
                  f"see {log_path}")
            raise optuna.TrialPruned()
        final = float(m.group(1))
        std = float(m.group(2))
        trial.set_user_attr("det_eval_std", std)
        print(f"[trial {trial.number}] det_eval mean = {final:.2f} "
              f"(std {std:.2f})")
        return final

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=80,
                        help="Sized so the sweep finishes inside ~1h at "
                             "n_jobs=6: at num_envs>=2 trials run roughly "
                             "3-15 min each on this box, with pruning killing "
                             "the slow / weak ones early.")
    parser.add_argument("--n-jobs", type=int, default=6,
                        help="Parallel trials. 32 cores / 6 jobs ~= 5 cores "
                             "per trial which leaves headroom for MuJoCo and "
                             "PyTorch threading.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=DEFAULT_STUDY)
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--n-startup-random", type=int, default=12)
    parser.add_argument("--prune-warmup-steps", type=int, default=100_000,
                        help="No pruning until each trial has reported past this step "
                             "(20%% of the 500k budget by default).")
    parser.add_argument("--prune-startup-trials", type=int, default=8,
                        help="Pruner waits this many completed trials before pruning. "
                             "Should be >= n_jobs so the first wave isn't pruned blind.")
    args = parser.parse_args()

    if shutil.which("uv") is None:
        raise SystemExit("`uv` not on PATH — the driver shells out to `uv run python`.")
    if not PPO_SCRIPT.exists():
        raise SystemExit(f"Expected ppo_continuous_action.py at {PPO_SCRIPT}")

    Path(args.storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=args.n_startup_random, seed=args.sampler_seed),
        pruner=MedianPruner(
            n_startup_trials=args.prune_startup_trials,
            n_warmup_steps=args.prune_warmup_steps,
            interval_steps=1,
        ),
        load_if_exists=True,
    )
    print(f"Study: {args.study_name} @ {args.storage}")
    print(f"Existing trials: {len(study.trials)}")
    print(f"Running {args.n_trials} new trials (n_jobs={args.n_jobs}, "
          f"budget per trial = {args.total_timesteps:,} steps).")

    study.optimize(make_objective(args), n_trials=args.n_trials, n_jobs=args.n_jobs,
                   gc_after_trial=True)

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"value (final return): {best.value:.2f}")
    print(f"params:")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    print(f"\nInspect with `optuna-dashboard {args.storage}` "
          f"(install via `uv tool install optuna-dashboard`).")


if __name__ == "__main__":
    main()

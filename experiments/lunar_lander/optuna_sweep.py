# /// script
# requires-python = ">=3.11"
# dependencies = ["optuna>=3.6"]
# ///
"""Optuna sweep over critical PPO hyperparameters on LunarLander-v3.

Each trial spawns `algos/ppo.py` as a subprocess (via `uv run`) with sampled
hparams and an `--optuna-report-path` that ppo.py uses to emit per-iteration
JSONL lines `{global_step, return, n_recent}`. The driver tails the file,
forwards values to Optuna, and terminates the subprocess on prune.

Default budget: 30 trials × up to 1M env steps each, with a MedianPruner
that starts pruning after the 5th trial and a 200k-step warmup. Results
persist to SQLite under `./optuna_studies/` so you can resume.

Run from this directory: `uv run optuna_sweep.py`.
"""

from __future__ import annotations

import argparse
import json
import os
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
PPO_SCRIPT = REPO_ROOT / "algos" / "ppo.py"

DEFAULT_STORAGE = f"sqlite:///{HERE / 'optuna_studies' / 'lander_ppo.db'}"
DEFAULT_STUDY = "lander_ppo_v1"
POLL_INTERVAL_S = 2.0
TERMINATE_GRACE_S = 30.0


def sample_params(trial: optuna.Trial) -> dict:
    """The critical PPO knob set.

    Choices follow Andrychowicz et al. 2021 ("What Matters in On-Policy RL?")
    and the cleanrl PPO implementation notes:
      * discount and GAE-λ are sampled log-uniformly on their complement
        (1-γ, 1-λ) so we get uniform coverage in effective-horizon space;
      * the high-importance knobs (γ, lr, λ, vf_coef) all participate;
      * minor knobs (max_grad_norm, norm_adv) are kept at defaults;
      * num_envs=16 is held fixed (matches optimal.sh).
    """
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 1e-3, 5e-2, log=True)
    one_minus_lambda = trial.suggest_float("one_minus_lambda", 1e-2, 1e-1, log=True)
    gamma = 1.0 - one_minus_gamma
    gae_lambda = 1.0 - one_minus_lambda
    # Persist the derived values for dashboard readability.
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("gae_lambda", gae_lambda)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "num_steps": trial.suggest_categorical("num_steps", [128, 256, 512, 1024]),
        "num_minibatches": trial.suggest_categorical("num_minibatches", [4, 8, 16, 32]),
        "update_epochs": trial.suggest_int("update_epochs", 3, 15),
        "clip_coef": trial.suggest_float("clip_coef", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.5),
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "anneal_lr": trial.suggest_categorical("anneal_lr", [True, False]),
    }


def build_cmd(params: dict, *, total_timesteps: int, seed: int, exp_name: str,
              report_path: Path) -> list[str]:
    cmd = [
        "uv", "run", "python", str(PPO_SCRIPT),
        "--env-id", "LunarLander-v3",
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
        "--num-envs", "16",
        "--exp-name", exp_name,
        "--no-capture-video",
        "--no-capture-test-video",
        "--optuna-report-path", str(report_path),
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
    ]
    return cmd


def consume_reports(fh, trial: optuna.Trial) -> bool:
    """Read all currently-available JSONL lines, call trial.report().

    Uses `readline()` in a loop so the file can be tailed across multiple
    polls — a `for line in fh` loop would exhaust the iterator at first EOF.
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
        # Send SIGTERM to the whole process group (uv may have spawned a child).
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
        # Pin torch / numpy / openblas to 1 intra-op thread per subprocess.
        # With n_jobs concurrent trials we'd otherwise oversubscribe the CPU.
        env = {**os.environ,
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        # New process group so we can SIGTERM the whole tree on prune.
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )

        # Tail report_path while the subprocess runs.
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
            # Drain any final lines.
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

        entries = read_all_reports(report_path)
        if not entries:
            print(f"[trial {trial.number}] no intermediate reports emitted; "
                  f"see {log_path}")
            raise optuna.TrialPruned()

        # Final score: mean of `return` over the last 10% of reports
        # (each report is itself a 100-episode rolling mean, so this is stable).
        k = max(1, int(0.1 * len(entries)))
        final = float(sum(e["return"] for e in entries[-k:]) / k)
        print(f"[trial {trial.number}] final return = {final:.2f}")
        return final

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=6,
                        help="Parallel trials. Each runs ppo.py on the GPU; tune to fit "
                             "VRAM and CPU. Default 6 sized for a 3090 + 4c/8t CPU.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study-name", default=DEFAULT_STUDY)
    parser.add_argument("--seed-base", type=int, default=1,
                        help="Trial seed = seed_base + trial.number.")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--n-startup-random", type=int, default=10,
                        help="TPE: random trials before model-based suggestions.")
    parser.add_argument("--prune-warmup-steps", type=int, default=200_000)
    parser.add_argument("--prune-startup-trials", type=int, default=10,
                        help="Pruner waits this many *completed* trials before pruning. "
                             "Should be >= n_jobs so the first wave isn't pruned blind.")
    args = parser.parse_args()

    if shutil.which("uv") is None:
        raise SystemExit("`uv` not on PATH — the driver shells out to `uv run python`.")
    if not PPO_SCRIPT.exists():
        raise SystemExit(f"Expected ppo.py at {PPO_SCRIPT}")

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
    print(f"Running {args.n_trials} new trials (n_jobs={args.n_jobs}).")

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

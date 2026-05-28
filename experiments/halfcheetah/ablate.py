"""Leave-one-out ablation around the winning HalfCheetah Optuna config,
plus head-to-head against published cleanrl & SB3-zoo configs.

Reads `reeval_results/winner.json` (produced by reeval_top.py) and:

  1. Runs the winner as `A_winner` (baseline).
  2. For each algorithmic knob where the winner deviates from the cleanrl
     defaults, flips that ONE knob back to the default. This tells us which
     of the optimizer's choices actually matter — if an arm matches the
     baseline within seed noise, that knob is incidental; if it collapses,
     that knob is load-bearing.
  3. Runs the full cleanrl defaults and the full SB3-zoo HalfCheetah-v4
     hyperparameter set as their own arms for direct comparison.

Rollout structure (num_envs, num_steps, num_minibatches) is ablated as a
single combined arm — those three interact strongly through the
batch_size = num_envs * num_steps / num_minibatches identity, so flipping
them one-at-a-time conflates several effects.

Score = DETERMINISTIC eval mean (action = policy mean, no exploration
noise), the SB3 metric. 3 seeds per arm by default.

Run: `uv run python ablate.py` (after reeval_top.py has populated
reeval_results/winner.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PPO_SCRIPT = REPO_ROOT / "algos" / "ppo_continuous_action.py"

DEFAULT_WINNER = HERE / "reeval_results" / "winner.json"
DEFAULT_OUT = HERE / "ablation_results"
DEFAULT_SEEDS = [201, 202, 203]   # disjoint from sweep (1..80) and reeval (101..105)
DEFAULT_TOTAL_STEPS = 500_000
DEFAULT_N_JOBS = 6

# cleanrl ppo_continuous_action.py shipped defaults (see algos/ppo_continuous_action.py).
CLEANRL_DEFAULTS: dict = {
    "learning_rate": 3e-4,
    "num_envs": 1,
    "num_steps": 2048,
    "num_minibatches": 32,
    "update_epochs": 10,
    "clip_coef": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "anneal_lr": True,
    "clip_vloss": True,
    "normalize": True,
}

# SB3-zoo HalfCheetah-v4 PPO config (hyperparams/ppo.yml in rl-baselines3-zoo).
# batch_size 64 with n_envs=1, n_steps=512 -> num_minibatches = 512/64 = 8.
# SB3 uses a linear LR schedule by default, matched by anneal_lr=True here.
# max_grad_norm is 0.8 upstream but this script's sweep holds it at the
# cleanrl default 0.5 (the algo script doesn't expose --max-grad-norm
# differently per arm), so SB3 arm matches that fixed value too.
SB3_HALFCHEETAH: dict = {
    "learning_rate": 2.0633e-5,
    "num_envs": 1,
    "num_steps": 512,
    "num_minibatches": 8,
    "update_epochs": 20,
    "clip_coef": 0.1,
    "ent_coef": 0.000401762,
    "vf_coef": 0.58096,
    "gamma": 0.98,
    "gae_lambda": 0.92,
    "anneal_lr": True,
    "clip_vloss": False,
    "normalize": True,
}

# Knobs ablated one-at-a-time (each is flipped from winner-value to CLEANRL_DEFAULTS).
# Rollout structure (num_envs/num_steps/num_minibatches) is ablated as one
# combined arm via ROLLOUT_KEYS below to avoid conflating three coupled knobs.
SCALAR_ABLATE_KEYS = [
    "learning_rate", "update_epochs", "clip_coef", "ent_coef", "vf_coef",
    "gamma", "gae_lambda",
]
TOGGLE_ABLATE_KEYS = ["anneal_lr", "clip_vloss", "normalize"]
ROLLOUT_KEYS = ["num_envs", "num_steps", "num_minibatches"]

DET_RE = re.compile(
    r"DET_EVAL episodes=(\d+) mean=(-?[\d.]+) std=(-?[\d.]+) "
    r"min=(-?[\d.]+) max=(-?[\d.]+)"
)

BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(0)


def params_to_cli(params: dict) -> list[str]:
    return [
        "--learning-rate", repr(float(params["learning_rate"])),
        "--num-envs", str(int(params["num_envs"])),
        "--num-steps", str(int(params["num_steps"])),
        "--num-minibatches", str(int(params["num_minibatches"])),
        "--update-epochs", str(int(params["update_epochs"])),
        "--clip-coef", repr(float(params["clip_coef"])),
        "--ent-coef", repr(float(params["ent_coef"])),
        "--vf-coef", repr(float(params["vf_coef"])),
        "--gae-lambda", repr(float(params["gae_lambda"])),
        "--gamma", repr(float(params["gamma"])),
        "--anneal-lr" if bool(params["anneal_lr"]) else "--no-anneal-lr",
        "--clip-vloss" if bool(params["clip_vloss"]) else "--no-clip-vloss",
        "--normalize" if bool(params["normalize"]) else "--no-normalize",
    ]


def build_cmd(params: dict, *, total_steps: int, seed: int, exp_name: str,
              report_path: Path) -> list[str]:
    return [
        "uv", "run", "python", str(PPO_SCRIPT),
        "--env-id", "HalfCheetah-v4",
        "--total-timesteps", str(total_steps),
        "--seed", str(seed),
        "--exp-name", exp_name,
        "--no-capture-video", "--no-capture-test-video",
        "--optuna-report-path", str(report_path),
        "--eval-episodes", "10",
        *params_to_cli(params),
    ]


def _near(winner_val, default_val, rel_tol: float = 0.10) -> bool:
    """True if winner value is within rel_tol (or exactly equal for bools).

    Skips ablation arms that would flip a knob to a near-identical value —
    e.g. winner gamma=0.9914 vs default 0.99 isn't meaningfully an
    ablation, just a re-run of the baseline."""
    if isinstance(winner_val, bool) or isinstance(default_val, bool):
        return winner_val == default_val
    try:
        wv = float(winner_val)
        dv = float(default_val)
    except (TypeError, ValueError):
        return winner_val == default_val
    if dv == 0.0:
        return abs(wv) < 1e-9
    return abs(wv - dv) / abs(dv) <= rel_tol


def make_arms(winner: dict) -> dict[str, dict]:
    """Build the arm dict {arm_name: full_param_dict}."""
    arms: dict[str, dict] = {}
    arms["A_winner"] = dict(winner)

    # Scalar / toggle leave-one-out arms.
    for key in SCALAR_ABLATE_KEYS + TOGGLE_ABLATE_KEYS:
        if _near(winner[key], CLEANRL_DEFAULTS[key]):
            continue   # winner value is already at/near default — no signal to gain
        arm = dict(winner)
        arm[key] = CLEANRL_DEFAULTS[key]
        arms[f"flip_{key}_to_default"] = arm

    # Rollout-structure combined arm.
    if any(winner[k] != CLEANRL_DEFAULTS[k] for k in ROLLOUT_KEYS):
        arm = dict(winner)
        for k in ROLLOUT_KEYS:
            arm[k] = CLEANRL_DEFAULTS[k]
        arms["flip_rollout_to_default"] = arm

    arms["Y_cleanrl_defaults"] = dict(CLEANRL_DEFAULTS)
    arms["Z_sb3_zoo_halfcheetah"] = dict(SB3_HALFCHEETAH)
    return arms


def read_jsonl(path: Path) -> list[dict]:
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


def parse_det_eval(log_path: Path) -> float:
    m = DET_RE.search(log_path.read_text())
    return float(m.group(2)) if m else float("nan")


def run_one(arm: str, params: dict, seed: int, *, total_steps: int,
            reports_dir: Path, log_dir: Path) -> dict:
    report_path = reports_dir / f"{arm}_s{seed}.jsonl"
    if report_path.exists():
        report_path.unlink()
    log_path = log_dir / f"{arm}_s{seed}.log"
    exp_name = f"abl-{arm}-s{seed}"
    cmd = build_cmd(params, total_steps=total_steps, seed=seed,
                    exp_name=exp_name, report_path=report_path)
    env = {**os.environ,
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    t0 = time.time()
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        rc = proc.wait()
    elapsed = time.time() - t0
    det = parse_det_eval(log_path)
    return {
        "arm": arm, "seed": seed, "rc": rc, "det": det,
        "elapsed_s": elapsed,
        "report_path": str(report_path),
        "log_path": str(log_path),
    }


def bootstrap_mean_ci(samples: np.ndarray) -> tuple[float, float, float]:
    s = np.asarray(samples, dtype=np.float64)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return (float("nan"),) * 3
    if s.size == 1:
        return float(s[0]), float(s[0]), float(s[0])
    boots = RNG.choice(s, size=(BOOTSTRAP_RESAMPLES, s.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(s.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-json", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.winner_json.exists():
        raise SystemExit(f"winner.json not found at {args.winner_json} — "
                         f"run reeval_top.py first.")
    if shutil.which("uv") is None:
        raise SystemExit("`uv` not on PATH (driver shells out to `uv run python`).")

    winner_meta = json.loads(args.winner_json.read_text())
    winner_params = winner_meta["params"]
    print(f"Winner: trial #{winner_meta['trial']}  reeval mean = "
          f"{winner_meta['mean']:.2f} "
          f"[{winner_meta['ci_lo']:.2f}, {winner_meta['ci_hi']:.2f}]")

    arms = make_arms(winner_params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.out_dir / "jsonl"
    reports_dir.mkdir(exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Persist the resolved arm specs for reproducibility / inspection.
    with (args.out_dir / "arms.json").open("w") as f:
        json.dump({k: v for k, v in arms.items()}, f, indent=2, default=float)

    print(f"\nArms ({len(arms)}):")
    for name in arms:
        diff_keys = [k for k in CLEANRL_DEFAULTS
                     if arms[name].get(k) != CLEANRL_DEFAULTS[k]]
        print(f"  {name:<32} differs from cleanrl on: {diff_keys}")

    jobs = [(arm, params, s) for arm, params in arms.items() for s in args.seeds]
    print(f"\nScheduling {len(jobs)} runs "
          f"({len(arms)} arms × {len(args.seeds)} seeds) "
          f"at {args.total_steps:,} steps each, n_jobs={args.n_jobs}.")

    results: list[dict] = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = [
            pool.submit(run_one, arm, params, s, total_steps=args.total_steps,
                        reports_dir=reports_dir, log_dir=log_dir)
            for (arm, params, s) in jobs
        ]
        for fut in as_completed(futures):
            r = fut.result()
            print(f"  done {r['arm']:<32} seed={r['seed']:>3} "
                  f"det={r['det']:>8.2f} rc={r['rc']} "
                  f"t={r['elapsed_s']:.0f}s")
            results.append(r)
    total_elapsed = time.time() - t_start
    print(f"\nAll runs complete. Wall time: {total_elapsed / 60:.1f} min")

    with (args.out_dir / "raw_runs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "det", "rc", "elapsed_s"])
        for r in results:
            w.writerow([r["arm"], r["seed"], f"{r['det']:.4f}", r["rc"],
                        f"{r['elapsed_s']:.1f}"])

    # --- Per-arm summary, ordered to match arms (so winner sits first).
    by_arm: dict[str, list[float]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r["det"])

    base_vals = np.array(by_arm.get("A_winner", []))
    base_mean = float(np.nanmean(base_vals)) if base_vals.size else float("nan")

    summary = []
    for arm in arms:
        vals = np.array(by_arm.get(arm, []))
        mean, lo, hi = bootstrap_mean_ci(vals)
        finite = vals[~np.isnan(vals)]
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        delta = mean - base_mean
        summary.append({
            "arm": arm, "n": int(finite.size), "mean": mean, "std": std,
            "ci_lo": lo, "ci_hi": hi, "delta_vs_winner": delta,
            "per_seed_vals": vals.tolist(),
        })

    print("\n=== Leave-one-out ablation (DET_EVAL, mean ± 95% bootstrap CI) ===")
    print(f"{'arm':<34} {'n':>2} {'mean':>9} {'95% CI':>22} {'std':>7} "
          f"{'Δ vs winner':>12}")
    print("-" * 92)
    for r in summary:
        tag = "  <-- baseline" if r["arm"] == "A_winner" else ""
        print(f"{r['arm']:<34} {r['n']:>2} {r['mean']:>9.2f} "
              f"[{r['ci_lo']:>8.2f},{r['ci_hi']:>8.2f}] {r['std']:>7.2f} "
              f"{r['delta_vs_winner']:>+12.2f}{tag}")

    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_seeds", "mean", "std", "ci_lo", "ci_hi",
                    "delta_vs_winner", "per_seed"])
        for r in summary:
            w.writerow([r["arm"], r["n"], f"{r['mean']:.4f}",
                        f"{r['std']:.4f}", f"{r['ci_lo']:.4f}",
                        f"{r['ci_hi']:.4f}", f"{r['delta_vs_winner']:.4f}",
                        ";".join(f"{v:.4f}" for v in r["per_seed_vals"])])

    # --- Horizontal bar chart: each arm with mean + 95% CI + per-seed dots.
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(summary) + 2.5))
    ys = np.arange(len(summary))
    means = [r["mean"] for r in summary]
    los = [r["ci_lo"] for r in summary]
    his = [r["ci_hi"] for r in summary]
    xerr = np.array([[m - lo for m, lo in zip(means, los)],
                     [hi - m for m, hi in zip(means, his)]])

    def colour(name: str):
        if name == "A_winner":
            return "#1f77b4"
        if name.startswith("Y_cleanrl"):
            return "#d62728"
        if name.startswith("Z_sb3"):
            return "#9467bd"
        return "#2ca02c"

    colors = [colour(r["arm"]) for r in summary]
    ax.barh(ys, means, xerr=xerr, capsize=4, color=colors, alpha=0.75,
            edgecolor="black", linewidth=0.5)
    for y, r in zip(ys, summary):
        for v in r["per_seed_vals"]:
            if np.isnan(v):
                continue
            ax.scatter(v, y + RNG.uniform(-0.12, 0.12), color="black", s=20, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["arm"] for r in summary])
    ax.invert_yaxis()
    ax.axvline(base_mean, color="grey", linestyle="--", linewidth=1,
               label=f"winner mean ({base_mean:.0f})")
    ax.set_xlabel("DET_EVAL mean return")
    ax.set_title("HalfCheetah PPO ablation — leave-one-out around the sweep winner,\n"
                 "plus cleanrl-defaults and SB3-zoo HalfCheetah-v4 baselines")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "ablation.png", dpi=160)
    plt.close(fig)
    print(f"\nWrote {args.out_dir/'summary.csv'} and {args.out_dir/'ablation.png'}")


if __name__ == "__main__":
    main()

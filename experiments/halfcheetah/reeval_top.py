"""Re-evaluate the top-K HalfCheetah Optuna trials at N fresh seeds.

Mirrors experiments/pendulum/reeval_top.py — reads top trials by final value
from a study in the Optuna SQLite store, spawns ppo_continuous_action.py for
each (trial, seed) combination, then ranks by seed-averaged final return
with 95% bootstrap CIs.

The HalfCheetah sweep adds `clip_vloss` and `normalize` to the search space
on top of what pendulum sweeps, so those flags are threaded through here.

Defaults: top 3 trials x 5 seeds = 15 runs, 500k steps each, 6 in parallel.
Run from this directory: `uv run python reeval_top.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PPO_SCRIPT = REPO_ROOT / "algos" / "ppo_continuous_action.py"

DEFAULT_DB = HERE / "optuna_studies" / "halfcheetah_ppo.db"
DEFAULT_STUDY = "halfcheetah_ppo_v1"
DEFAULT_OUT = HERE / "reeval_results"
DEFAULT_SEEDS = [101, 102, 103, 104, 105]  # disjoint from sweep seeds (1..80)
DEFAULT_K = 3
DEFAULT_TOTAL_STEPS = 500_000
N_JOBS_DEFAULT = 6
# Reference: SB3-zoo HalfCheetah-v4 PPO published episode reward ~ 1976
# (rl-baselines3-zoo benchmark). Used as a horizontal reference line.
GOOD_RETURN = 2000.0
BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(0)


@dataclass
class TrialSpec:
    number: int
    sweep_value: float
    params: dict


def fetch_top_trials(db_path: Path, study_name: str, k: int) -> list[TrialSpec]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    study_row = con.execute(
        "SELECT study_id FROM studies WHERE study_name=?", (study_name,)
    ).fetchone()
    if study_row is None:
        con.close()
        raise SystemExit(f"Study {study_name!r} not found in {db_path}")
    study_id = study_row["study_id"]
    rows = list(con.execute("""
        SELECT t.trial_id, t.number, tv.value
        FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
        WHERE t.study_id=? AND t.state='COMPLETE'
        ORDER BY tv.value DESC LIMIT ?
    """, (study_id, k)))
    out: list[TrialSpec] = []
    for r in rows:
        params: dict = {}
        for p in con.execute(
            "SELECT param_name, param_value, distribution_json "
            "FROM trial_params WHERE trial_id=?", (r["trial_id"],)
        ):
            dist = json.loads(p["distribution_json"])
            name = p["param_name"]
            val = p["param_value"]
            if dist.get("name") == "CategoricalDistribution":
                params[name] = dist["attributes"]["choices"][int(val)]
            else:
                params[name] = val
        if "one_minus_gamma" in params:
            params["gamma"] = 1.0 - params.pop("one_minus_gamma")
        if "one_minus_lambda" in params:
            params["gae_lambda"] = 1.0 - params.pop("one_minus_lambda")
        out.append(TrialSpec(number=r["number"], sweep_value=r["value"], params=params))
    con.close()
    return out


def build_cmd(params: dict, *, total_steps: int, seed: int, exp_name: str,
              report_path: Path) -> list[str]:
    return [
        "uv", "run", "python", str(PPO_SCRIPT),
        "--env-id", "HalfCheetah-v4",
        "--total-timesteps", str(total_steps),
        "--seed", str(seed),
        "--num-envs", str(int(params["num_envs"])),
        "--exp-name", exp_name,
        "--no-capture-video", "--no-capture-test-video",
        "--optuna-report-path", str(report_path),
        "--eval-episodes", "10",
        "--learning-rate", repr(float(params["learning_rate"])),
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


def final_score(entries: list[dict], frac: float = 0.1) -> float:
    if not entries:
        return float("nan")
    k = max(1, int(frac * len(entries)))
    return float(sum(e["return"] for e in entries[-k:]) / k)


def run_one(trial: TrialSpec, seed: int, *, total_steps: int,
            reports_dir: Path, log_dir: Path) -> dict:
    report_path = reports_dir / f"trial_{trial.number:04d}_seed_{seed}.jsonl"
    if report_path.exists():
        report_path.unlink()
    exp_name = f"reeval-t{trial.number:04d}-s{seed}"
    cmd = build_cmd(trial.params, total_steps=total_steps, seed=seed,
                    exp_name=exp_name, report_path=report_path)
    log_path = log_dir / f"t{trial.number:04d}_s{seed}.log"
    env = {**os.environ,
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    t0 = time.time()
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        rc = proc.wait()
    elapsed = time.time() - t0
    entries = read_jsonl(report_path)
    return {
        "trial": trial.number,
        "seed": seed,
        "final": final_score(entries),
        "rc": rc,
        "elapsed_s": elapsed,
        "n_reports": len(entries),
        "report_path": str(report_path),
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


def per_seed_curves(results: list[dict], total_steps: int,
                    n_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, total_steps, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    curves: list[np.ndarray] = []
    for r in results:
        entries = read_jsonl(Path(r["report_path"]))
        out = np.full(n_bins, np.nan)
        if entries:
            steps = np.array([int(e["global_step"]) for e in entries])
            vals = np.array([float(e["return"]) for e in entries])
            idx = np.digitize(steps, edges) - 1
            for b in range(n_bins):
                mask = idx == b
                if mask.any():
                    out[b] = vals[mask].mean()
            last = np.nan
            for i in range(n_bins):
                if np.isnan(out[i]):
                    out[i] = last
                else:
                    last = out[i]
        curves.append(out)
    return centers, np.stack(curves, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--study-name", default=DEFAULT_STUDY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_K)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Optuna DB not found at {args.db}")
    if shutil.which("uv") is None:
        raise SystemExit("`uv` not on PATH (driver shells out to `uv run python`).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.out_dir / "jsonl"
    reports_dir.mkdir(exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    trials = fetch_top_trials(args.db, args.study_name, args.top_k)
    if not trials:
        raise SystemExit(f"No completed trials in study {args.study_name!r}.")
    print(f"Study {args.study_name!r} — top {len(trials)} completed trials:")
    for t in trials:
        print(f"  #{t.number:>3}   sweep value = {t.sweep_value:>9.2f}  "
              f"(num_envs={t.params.get('num_envs')})")

    jobs = [(t, s) for t in trials for s in args.seeds]
    print(f"\nScheduling {len(jobs)} runs "
          f"({len(trials)} trials × {len(args.seeds)} seeds) "
          f"at {args.total_steps:,} steps each, n_jobs={args.n_jobs}.")

    results: list[dict] = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = [
            pool.submit(run_one, t, s, total_steps=args.total_steps,
                        reports_dir=reports_dir, log_dir=log_dir)
            for t, s in jobs
        ]
        for fut in as_completed(futures):
            r = fut.result()
            print(f"  done: #{r['trial']:>3}  seed={r['seed']:>3}  "
                  f"final={r['final']:>9.2f}  rc={r['rc']}  "
                  f"t={r['elapsed_s']:.0f}s  n_reports={r['n_reports']}")
            results.append(r)
    total_elapsed = time.time() - t_start
    print(f"\nAll runs complete. Wall time: {total_elapsed / 60:.1f} min")

    with (args.out_dir / "raw_runs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "seed", "final", "rc", "elapsed_s", "n_reports",
                    "report_path"])
        for r in results:
            w.writerow([r["trial"], r["seed"], f"{r['final']:.4f}", r["rc"],
                        f"{r['elapsed_s']:.1f}", r["n_reports"], r["report_path"]])

    by_trial: dict[int, list[dict]] = {}
    for r in results:
        by_trial.setdefault(r["trial"], []).append(r)

    summary = []
    for t in trials:
        rs = sorted(by_trial.get(t.number, []), key=lambda r: r["seed"])
        vals = np.array([r["final"] for r in rs])
        mean, lo, hi = bootstrap_mean_ci(vals)
        finite = vals[~np.isnan(vals)]
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        sem = std / np.sqrt(finite.size) if finite.size else 0.0
        summary.append({
            "trial": t.number, "sweep_value": t.sweep_value,
            "n_seeds": int(finite.size),
            "mean": mean, "std": std, "sem": sem,
            "ci_lo": lo, "ci_hi": hi,
            "per_seed_vals": vals.tolist(),
            "per_seed_results": rs,
            "params": t.params,
        })
    summary.sort(key=lambda r: -r["mean"])

    print("\n=== Multi-seed ranking (mean ± 95% bootstrap CI over seeds) ===")
    print(f"{'rank':>4}  {'#trial':>6}  {'n':>3}  {'mean':>9}  {'95% CI':>22}"
          f"  {'std':>7}  {'sweep':>9}")
    for rank, r in enumerate(summary, 1):
        print(f"  {rank:>2}    #{r['trial']:>3}   {r['n_seeds']:>3}  "
              f"{r['mean']:>9.2f}  [{r['ci_lo']:>8.2f},{r['ci_hi']:>8.2f}]  "
              f"{r['std']:>7.2f}  {r['sweep_value']:>9.2f}")

    with (args.out_dir / "ranking.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "trial", "n_seeds", "mean", "std", "sem",
                    "ci_lo", "ci_hi", "sweep_value", "per_seed"])
        for rank, r in enumerate(summary, 1):
            w.writerow([rank, r["trial"], r["n_seeds"],
                        f"{r['mean']:.4f}", f"{r['std']:.4f}", f"{r['sem']:.4f}",
                        f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                        f"{r['sweep_value']:.4f}",
                        ";".join(f"{v:.4f}" for v in r["per_seed_vals"])])

    # --- Bar chart with seed dots and original sweep value as a red ×.
    fig, ax = plt.subplots(figsize=(1.9 * len(summary) + 2.5, 5.0))
    xs = np.arange(len(summary))
    means = [r["mean"] for r in summary]
    los = [r["ci_lo"] for r in summary]
    his = [r["ci_hi"] for r in summary]
    yerr = np.array([[m - lo for m, lo in zip(means, los)],
                     [hi - m for m, hi in zip(means, his)]])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))
    ax.bar(xs, means, yerr=yerr, capsize=6, color=colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    seen_label = False
    for x, r in zip(xs, summary):
        for v in r["per_seed_vals"]:
            if np.isnan(v):
                continue
            ax.scatter(x + RNG.uniform(-0.10, 0.10), v, color="black", s=24, zorder=5)
        lbl = "original sweep value" if not seen_label else None
        ax.scatter([x], [r["sweep_value"]], marker="x", color="red", s=55, zorder=6,
                   label=lbl)
        seen_label = True
    ax.set_xticks(xs)
    ax.set_xticklabels([f"#{r['trial']}\nenvs={r['params'].get('num_envs')}"
                        for r in summary])
    ax.set_ylabel("final return (mean of last 10% of training)")
    ax.set_title("HalfCheetah multi-seed re-evaluation of top trials\n"
                 "bars = mean ± 95% bootstrap CI · dots = seed scores · × = original sweep")
    ax.axhline(GOOD_RETURN, color="grey", linestyle="--", linewidth=1,
               label=f"SB3-zoo ref (~{GOOD_RETURN:.0f})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "ranking.png", dpi=160)
    plt.close(fig)

    # --- Learning curves overlay.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for r, color in zip(summary, colors):
        centers, curves = per_seed_curves(r["per_seed_results"], args.total_steps)
        n_bins = curves.shape[1]
        mean = np.full(n_bins, np.nan)
        lo = np.full(n_bins, np.nan)
        hi = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m, l, h = bootstrap_mean_ci(curves[:, b])
            mean[b], lo[b], hi[b] = m, l, h
        ax.plot(centers, mean, color=color, linewidth=2,
                label=f"#{r['trial']} (envs={r['params'].get('num_envs')})  "
                      f"final={r['mean']:.0f}")
        ax.fill_between(centers, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.axhline(GOOD_RETURN, color="grey", linestyle="--", linewidth=1,
               label=f"SB3-zoo ref (~{GOOD_RETURN:.0f})")
    ax.set_xlabel("environment step")
    ax.set_ylabel("episodic return (mean ± 95% CI across seeds)")
    ax.set_title(f"HalfCheetah top-{len(trials)} re-evaluation learning curves "
                 f"({len(args.seeds)} seeds each)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    winner = summary[0]
    winner_path = args.out_dir / "winner.json"
    with winner_path.open("w") as f:
        json.dump({
            "study": args.study_name,
            "trial": winner["trial"],
            "mean": winner["mean"],
            "ci_lo": winner["ci_lo"],
            "ci_hi": winner["ci_hi"],
            "std": winner["std"],
            "n_seeds": winner["n_seeds"],
            "params": winner["params"],
            "per_seed_vals": winner["per_seed_vals"],
            "seeds_used": args.seeds,
        }, f, indent=2)

    print(f"\nWrote {args.out_dir / 'ranking.csv'}, "
          f"{args.out_dir / 'ranking.png'}, "
          f"{args.out_dir / 'learning_curves.png'}, {winner_path}")
    print(f"\n=== Winner: trial #{winner['trial']}  "
          f"mean = {winner['mean']:.2f}  "
          f"[{winner['ci_lo']:.2f}, {winner['ci_hi']:.2f}]  "
          f"std = {winner['std']:.2f}   "
          f"(sweep value was {winner['sweep_value']:.2f}) ===")


if __name__ == "__main__":
    main()

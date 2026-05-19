# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///
"""One-at-a-time ablation around the multi-seed winner.

Starts from `reeval_results/winner.json` and, for each PPO hyperparameter,
overrides it with a "sensible alternative" (the cleanrl default, the value
used in `optimal.sh`, or the opposite end of the searched range). Each
perturbed config is run at N seeds (default 3, matched to a subset of the
re-evaluation seeds), and the drop relative to the multi-seed baseline is
reported.

The 'importance' here is the partial sensitivity of the score to each knob
around the winner's basin — not a full-variance decomposition. With ~3 seeds
per perturbation, differences smaller than ~30 reward units are noise.

Run from this directory: `uv run ablate.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PPO_SCRIPT = REPO_ROOT / "algos" / "ppo.py"

DEFAULT_WINNER = HERE / "reeval_results" / "winner.json"
DEFAULT_REEVAL_RAW = HERE / "reeval_results" / "raw_runs.csv"
DEFAULT_OUT = HERE / "ablation_results"
DEFAULT_SEEDS = [101, 102, 103]   # subset of re-eval seeds → matched comparison
DEFAULT_TOTAL_STEPS = 1_000_000
N_JOBS_DEFAULT = 6
BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(0)


def make_perturbations(winner: dict) -> list[tuple[str, object, str]]:
    """(param_name, alt_value, short_label). The label says what we changed it *to*."""
    return [
        ("learning_rate",    3e-4,   "lr 3e-4 (cleanrl default)"),
        ("num_steps",        1024,   "num_steps 1024 (large rollouts)"),
        ("num_minibatches",  32,     "num_minibatches 32 (smaller mb)"),
        ("update_epochs",    4,      "update_epochs 4 (cleanrl default)"),
        ("clip_coef",        0.20,   "clip_coef 0.20 (PPO paper default)"),
        ("ent_coef",         0.02,   "ent_coef 0.02 (optimal.sh value)"),
        ("vf_coef",          0.50,   "vf_coef 0.50 (cleanrl default)"),
        ("gamma",            0.99,   "gamma 0.99 (standard default)"),
        ("gae_lambda",       0.95,   "gae_lambda 0.95 (cleanrl default)"),
        ("anneal_lr",        True,   "anneal_lr ON"),
    ]


@dataclass
class Run:
    name: str             # short label for this ablation cell
    param: str            # which knob was perturbed
    value: object         # what it was set to
    seed: int
    final: float
    rc: int
    elapsed_s: float
    n_reports: int
    report_path: str


def build_cmd(params: dict, *, total_steps: int, seed: int, exp_name: str,
              report_path: Path) -> list[str]:
    return [
        "uv", "run", "python", str(PPO_SCRIPT),
        "--env-id", "LunarLander-v3",
        "--total-timesteps", str(total_steps),
        "--seed", str(seed),
        "--num-envs", "16",
        "--exp-name", exp_name,
        "--no-capture-video", "--no-capture-test-video",
        "--optuna-report-path", str(report_path),
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


def run_one(params: dict, *, label: str, param: str, value, seed: int,
            total_steps: int, reports_dir: Path, log_dir: Path) -> Run:
    safe_label = label.split()[0].replace("/", "_")
    report_path = reports_dir / f"{safe_label}_seed_{seed}.jsonl"
    if report_path.exists():
        report_path.unlink()
    exp_name = f"ablate-{safe_label}-s{seed}"
    cmd = build_cmd(params, total_steps=total_steps, seed=seed,
                    exp_name=exp_name, report_path=report_path)
    log_path = log_dir / f"{safe_label}_seed_{seed}.log"
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
    return Run(
        name=label, param=param, value=value, seed=seed,
        final=final_score(entries), rc=rc, elapsed_s=elapsed,
        n_reports=len(entries), report_path=str(report_path),
    )


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


def baseline_from_reeval(raw_csv: Path, winner_trial: int, seeds: list[int]) -> dict:
    """Pull the matched-seed subset of the multi-seed re-eval for the winning trial."""
    out: dict[int, float] = {}
    with raw_csv.open() as f:
        for r in csv.DictReader(f):
            if int(r["trial"]) != winner_trial:
                continue
            seed = int(r["seed"])
            if seed in seeds:
                out[seed] = float(r["final"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--reeval-csv", type=Path, default=DEFAULT_REEVAL_RAW)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.winner.exists():
        raise SystemExit(f"Winner JSON not found at {args.winner} — run reeval_top.py first.")
    if shutil.which("uv") is None:
        raise SystemExit("`uv` not on PATH (driver shells out to `uv run python`).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.out_dir / "jsonl"
    reports_dir.mkdir(exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    winner_obj = json.loads(args.winner.read_text())
    winner_params = winner_obj["params"]
    winner_trial = int(winner_obj["trial"])
    baseline_at_seeds = baseline_from_reeval(args.reeval_csv, winner_trial, args.seeds)
    baseline_all = np.array(winner_obj["per_seed_vals"])

    print(f"Baseline (winner trial #{winner_trial}):")
    print(f"  multi-seed mean over all 5 reeval seeds: {np.nanmean(baseline_all):.2f}")
    if baseline_at_seeds:
        seed_vals = np.array([baseline_at_seeds[s] for s in args.seeds if s in baseline_at_seeds])
        print(f"  matched-seed mean over seeds {sorted(baseline_at_seeds)}: "
              f"{seed_vals.mean():.2f}")

    perturbations = make_perturbations(winner_params)
    print(f"\nScheduling {len(perturbations) * len(args.seeds)} runs "
          f"({len(perturbations)} perturbations × {len(args.seeds)} seeds), "
          f"n_jobs={args.n_jobs}.")

    jobs: list[tuple[dict, str, str, object, int]] = []
    for param, value, label in perturbations:
        params = dict(winner_params)
        params[param] = value
        for seed in args.seeds:
            jobs.append((params, label, param, value, seed))

    results: list[Run] = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = [pool.submit(run_one, p, label=lbl, param=par, value=v, seed=s,
                               total_steps=args.total_steps,
                               reports_dir=reports_dir, log_dir=log_dir)
                   for (p, lbl, par, v, s) in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            print(f"  done: {r.name[:42]:>42}  seed={r.seed:>3}  "
                  f"final={r.final:>7.2f}  rc={r.rc}  t={r.elapsed_s:.0f}s")
            results.append(r)
    total_elapsed = time.time() - t_start
    print(f"\nAll runs complete. Wall time: {total_elapsed / 60:.1f} min")

    # Group results by perturbation label.
    by_label: dict[str, list[Run]] = {}
    for r in results:
        by_label.setdefault(r.name, []).append(r)

    base_matched = np.array([baseline_at_seeds[s] for s in args.seeds
                             if s in baseline_at_seeds])
    base_mean, base_lo, base_hi = bootstrap_mean_ci(baseline_all)

    summary = []
    for param, value, label in perturbations:
        rs = sorted(by_label.get(label, []), key=lambda r: r.seed)
        vals = np.array([r.final for r in rs])
        mean, lo, hi = bootstrap_mean_ci(vals)
        # Paired drops (per-seed), where matched baseline is available.
        paired = np.array([
            (baseline_at_seeds[r.seed] - r.final)
            for r in rs if r.seed in baseline_at_seeds and not np.isnan(r.final)
        ])
        pmean, plo, phi = bootstrap_mean_ci(paired)
        summary.append({
            "param": param, "value": value, "label": label,
            "n_seeds": int(np.sum(~np.isnan(vals))),
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "drop_vs_baseline_mean": base_mean - mean,
            "paired_drop_mean": pmean,
            "paired_drop_ci_lo": plo, "paired_drop_ci_hi": phi,
            "per_seed_vals": vals.tolist(),
            "per_seed_results": rs,
        })

    # Rank by paired drop (largest drop = most important knob).
    summary.sort(key=lambda r: -r["paired_drop_mean"])

    print(f"\nBaseline (winner trial #{winner_trial}, all 5 reeval seeds): "
          f"{base_mean:.2f}  [{base_lo:.2f}, {base_hi:.2f}]\n")
    print(f"{'rank':>4}  {'param':>16}  {'→ alt value':<30}  "
          f"{'perturbed mean':>14}  {'paired Δ (CI)':>22}")
    print("-" * 100)
    for rank, r in enumerate(summary, 1):
        val_str = str(r["value"])
        lo_str = f"{r['paired_drop_ci_lo']:>5.1f}"
        hi_str = f"{r['paired_drop_ci_hi']:>5.1f}"
        print(f"  {rank:>2}  {r['param']:>16}  {val_str:<30}  "
              f"{r['mean']:>14.2f}  "
              f"{r['paired_drop_mean']:>7.2f}  [{lo_str},{hi_str}]")

    # CSV.
    with (args.out_dir / "ranking.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "param", "alt_value", "label", "n_seeds",
                    "perturbed_mean", "perturbed_ci_lo", "perturbed_ci_hi",
                    "drop_vs_baseline_mean",
                    "paired_drop_mean", "paired_drop_ci_lo", "paired_drop_ci_hi",
                    "per_seed_vals"])
        for rank, r in enumerate(summary, 1):
            w.writerow([rank, r["param"], r["value"], r["label"], r["n_seeds"],
                        f"{r['mean']:.4f}", f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                        f"{r['drop_vs_baseline_mean']:.4f}",
                        f"{r['paired_drop_mean']:.4f}",
                        f"{r['paired_drop_ci_lo']:.4f}",
                        f"{r['paired_drop_ci_hi']:.4f}",
                        ";".join(f"{v:.4f}" for v in r["per_seed_vals"])])

    # --- Bar chart of paired drops, ranked.
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ys = np.arange(len(summary))
    drops = [r["paired_drop_mean"] for r in summary]
    los = [r["paired_drop_ci_lo"] for r in summary]
    his = [r["paired_drop_ci_hi"] for r in summary]
    xerr = np.array([[d - lo for d, lo in zip(drops, los)],
                     [hi - d for d, hi in zip(drops, his)]])
    colors = plt.cm.RdYlGn_r((np.array(drops) - min(drops)) /
                              max(1e-9, max(drops) - min(drops)))
    ax.barh(ys, drops, xerr=xerr, color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5)
    for y, r in zip(ys, summary):
        # Per-seed drop dots (one per seed for which we have matched baseline).
        for rr in r["per_seed_results"]:
            if rr.seed in baseline_at_seeds and not np.isnan(rr.final):
                d = baseline_at_seeds[rr.seed] - rr.final
                ax.scatter(d, y + RNG.uniform(-0.10, 0.10), color="black",
                           s=22, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in summary])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("paired drop vs winner baseline  (positive = worse than tuned value)")
    ax.set_title(
        f"One-at-a-time ablation around trial #{winner_trial}\n"
        f"baseline = {base_mean:.0f} · {len(args.seeds)} matched seeds per perturbation"
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "ranking.png", dpi=160)
    plt.close(fig)

    # --- Per-perturbation final value bar chart with baseline line.
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(summary) + 2.5), 5.0))
    xs = np.arange(len(summary))
    means = [r["mean"] for r in summary]
    yerr = np.array([[m - r["ci_lo"] for m, r in zip(means, summary)],
                     [r["ci_hi"] - m for m, r in zip(means, summary)]])
    ax.bar(xs, means, yerr=yerr, capsize=5, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    for x, r in zip(xs, summary):
        for v in r["per_seed_vals"]:
            if np.isnan(v):
                continue
            ax.scatter(x + RNG.uniform(-0.10, 0.10), v, color="black", s=22, zorder=5)
    ax.axhline(base_mean, color="red", linewidth=1.5, linestyle="--",
               label=f"baseline (tuned) = {base_mean:.0f}")
    ax.fill_between([-0.5, len(summary) - 0.5], base_lo, base_hi,
                    color="red", alpha=0.10, label="baseline 95% CI")
    ax.axhline(200, color="grey", linestyle=":", linewidth=1, label="solved (200)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in summary], rotation=30, ha="right",
                        fontsize=8)
    ax.set_ylabel("final return (mean of last 10% of training)")
    ax.set_title(f"Each perturbation's final return vs the tuned baseline "
                 f"(trial #{winner_trial})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "final_values.png", dpi=160)
    plt.close(fig)

    print(f"\nWrote {args.out_dir / 'ranking.csv'}, "
          f"{args.out_dir / 'ranking.png'}, "
          f"{args.out_dir / 'final_values.png'}")


if __name__ == "__main__":
    main()

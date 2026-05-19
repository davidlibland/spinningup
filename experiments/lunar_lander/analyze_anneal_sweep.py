# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11", "tensorboard>=2.18"]
# ///
"""Analyze the annealed-LR num_steps sweep on LunarLander.

Scope: runs with exp_name matching `ppo-lr*-nsteps{N}-anneal` — 3 num_steps
values × 3 seeds = 9 runs at lr=3e-4, anneal_lr=True, update_epochs=10,
total_timesteps=800k.

Produces learning curves with 95% bootstrap CI across seeds, a final-
performance bar chart, and pairwise Welch's t-tests (Holm-corrected).

Run from this directory: `uv run analyze_anneal_sweep.py`.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from _tblog import RunData, bin_returns, dedupe_by, discover_all

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
OUT_DIR = HERE / "analysis_anneal_sweep"

N_BINS = 80
FINAL_FRAC = 0.1
BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(0)


def select_sweep_runs(all_runs: list[RunData]) -> list[RunData]:
    """Pick the new sweep: anneal_lr=True, update_epochs=10, lr=3e-4, T=800k."""
    out: list[RunData] = []
    for r in all_runs:
        if not r.anneal_lr:
            continue
        if r.update_epochs != 10:
            continue
        if abs(r.learning_rate - 3e-4) > 1e-9:
            continue
        if r.total_timesteps != 800_000:
            continue
        if "nsteps" not in r.exp_name or "anneal" not in r.exp_name:
            continue
        out.append(r)
    return out


def bootstrap_mean_ci(samples: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    s = np.asarray(samples, dtype=np.float64)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return (float("nan"),) * 3
    if s.size == 1:
        return float(s[0]), float(s[0]), float(s[0])
    boots = RNG.choice(s, size=(n_resamples, s.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(s.mean()), float(lo), float(hi)


def curve_ci(seed_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_seeds, n_bins = seed_curves.shape
    mean = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m, l, h = bootstrap_mean_ci(seed_curves[:, b])
        mean[b], lo[b], hi[b] = m, l, h
    return mean, lo, hi


def final_perf_per_seed(seed_curves: np.ndarray, frac: float = FINAL_FRAC) -> np.ndarray:
    k = max(1, int(round(frac * seed_curves.shape[1])))
    return np.nanmean(seed_curves[:, -k:], axis=1)


def holm_correct(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pvals[i] * (n - rank))
        adj[i] = min(running, 1.0)
    return adj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--total-steps", type=int, default=800_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = discover_all(args.runs_dir)
    sweep = dedupe_by(select_sweep_runs(all_runs), key=lambda r: (r.num_steps, r.seed))
    if not sweep:
        raise SystemExit("No runs from the annealed sweep found.")

    print(f"Loaded {len(sweep)} runs (expected 9).")
    by_ns: dict[int, list[RunData]] = defaultdict(list)
    for r in sweep:
        by_ns[r.num_steps].append(r)
    nstepss = sorted(by_ns)
    for ns in nstepss:
        seeds = sorted(r.seed for r in by_ns[ns])
        print(f"  num_steps={ns}: seeds {seeds}")

    edges = np.linspace(0, args.total_steps, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned: dict[int, np.ndarray] = {}
    for ns in nstepss:
        rs = sorted(by_ns[ns], key=lambda r: r.seed)
        binned[ns] = np.stack([bin_returns(r, edges) for r in rs], axis=0)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(nstepss)))

    # ---- Figure 1: learning curves with 95% bootstrap CI ----
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for ns, color in zip(nstepss, colors):
        m, lo, hi = curve_ci(binned[ns])
        ax.plot(centers, m, color=color, linewidth=2,
                label=f"num_steps = {ns} (n={binned[ns].shape[0]})")
        ax.fill_between(centers, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.axhline(200.0, color="grey", linestyle=":", linewidth=1, label="solved (200)")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("environment step")
    ax.set_ylabel("episodic return (mean across seeds)")
    ax.set_title("LunarLander-v3 / PPO — annealed-LR num_steps sweep "
                 "(lr=3e-4, anneal=Y, epochs=10, 800k steps, 3 seeds, 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(f"\nWrote {args.out_dir / 'learning_curves.png'}")

    # ---- Figure 2: per-seed small multiples ----
    fig, axes = plt.subplots(1, len(nstepss), figsize=(5.2 * len(nstepss), 4.2),
                              sharey=True)
    if len(nstepss) == 1:
        axes = [axes]
    for ax, ns, color in zip(axes, nstepss, colors):
        for i in range(binned[ns].shape[0]):
            ax.plot(centers, binned[ns][i], alpha=0.75, linewidth=1.2)
        ax.set_title(f"num_steps = {ns}")
        ax.set_xlabel("environment step")
        ax.axhline(200.0, color="grey", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("episodic return")
    fig.suptitle("Per-seed learning curves")
    fig.tight_layout()
    fig.savefig(args.out_dir / "per_seed_curves.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'per_seed_curves.png'}")

    # ---- Final-performance summary ----
    finals: dict[int, np.ndarray] = {ns: final_perf_per_seed(binned[ns]) for ns in nstepss}
    print(f"\nFinal performance (mean over last {int(FINAL_FRAC * 100)}% of training):")
    print(f"{'num_steps':>10}  {'n':>3}  {'mean':>8}  {'std':>8}  {'sem':>8}"
          f"  {'95% CI (bootstrap)':>26}")
    summary_rows = []
    for ns in nstepss:
        v = finals[ns]
        m, lo, hi = bootstrap_mean_ci(v)
        std = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        sem = std / np.sqrt(v.size) if v.size > 0 else 0.0
        print(f"{ns:>10}  {v.size:>3}  {m:>8.2f}  {std:>8.2f}  {sem:>8.2f}"
              f"  [{lo:>7.2f}, {hi:>7.2f}]")
        summary_rows.append((ns, v.size, m, std, sem, lo, hi))

    # ---- Welch's t-tests, Holm-corrected ----
    pairs = list(combinations(nstepss, 2))
    raw_p, tstats, diffs = [], [], []
    for a, b in pairs:
        t = stats.ttest_ind(finals[a], finals[b], equal_var=False, nan_policy="omit")
        raw_p.append(float(t.pvalue))
        tstats.append(float(t.statistic))
        diffs.append(float(np.nanmean(finals[a]) - np.nanmean(finals[b])))
    adj_p = holm_correct(raw_p)
    print("\nPairwise Welch's t-tests on final performance (Holm-corrected):")
    print(f"{'ns_a':>6}  {'ns_b':>6}  {'mean_a - mean_b':>16}  {'t':>7}  {'p':>9}  {'p_adj':>9}")
    pair_rows = []
    for (a, b), t, d, p, pa in zip(pairs, tstats, diffs, raw_p, adj_p):
        print(f"{a:>6}  {b:>6}  {d:>16.2f}  {t:>7.2f}  {p:>9.4f}  {pa:>9.4f}")
        pair_rows.append((a, b, d, t, p, pa))

    # ---- Figure 3: final-performance bar chart with seed dots ----
    fig, ax = plt.subplots(figsize=(1.8 * len(nstepss) + 2.0, 4.6))
    xs = np.arange(len(nstepss))
    means = [bootstrap_mean_ci(finals[ns])[0] for ns in nstepss]
    los = [bootstrap_mean_ci(finals[ns])[1] for ns in nstepss]
    his = [bootstrap_mean_ci(finals[ns])[2] for ns in nstepss]
    yerr = np.array([[m - lo for m, lo in zip(means, los)],
                     [hi - m for m, hi in zip(means, his)]])
    ax.bar(xs, means, yerr=yerr, capsize=6,
           color=colors, alpha=0.75, edgecolor="black", linewidth=0.5)
    for x, ns in zip(xs, nstepss):
        v = finals[ns]
        ax.scatter(np.full_like(v, x) + RNG.uniform(-0.08, 0.08, size=v.size),
                   v, color="black", s=24, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ns={ns}" for ns in nstepss])
    ax.set_ylabel(f"mean episodic return (last {int(FINAL_FRAC * 100)}% of training)")
    ax.set_title("Final performance per num_steps (95% bootstrap CI, dots = seeds)")
    ax.axhline(200.0, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "final_performance.png", dpi=160)
    plt.close(fig)
    print(f"\nWrote {args.out_dir / 'final_performance.png'}")

    # ---- CSV outputs ----
    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_steps", "n_seeds", "mean", "std", "sem", "ci95_lo", "ci95_hi"])
        w.writerows(summary_rows)
    with (args.out_dir / "pairwise.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ns_a", "ns_b", "mean_a_minus_b", "t_stat", "p_raw", "p_holm"])
        w.writerows(pair_rows)
    with (args.out_dir / "per_seed_final.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_steps", "seed", "final_mean_return"])
        for ns in nstepss:
            for r, val in zip(sorted(by_ns[ns], key=lambda r: r.seed), finals[ns]):
                w.writerow([ns, r.seed, f"{val:.4f}"])
    print(f"Wrote summary.csv, pairwise.csv, per_seed_final.csv to {args.out_dir}")


if __name__ == "__main__":
    main()

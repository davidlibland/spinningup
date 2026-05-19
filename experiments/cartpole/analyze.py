# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11", "tensorboard>=2.18"]
# ///
"""Statistical analysis of CartPole PPO runs across learning rates and seeds.

Reads TensorBoard event files from ./runs/, groups runs by learning rate, and
produces training curves with bootstrap 95% CIs across seeds plus a final-
performance summary with pairwise Welch's t-tests (Holm-corrected).

Run from this directory: `uv run analyze.py`.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
OUT_DIR = HERE / "analysis"

METRIC = "charts/episodic_return"
HPARAMS_TAG = "hyperparameters/text_summary"
N_BINS = 50
FINAL_FRAC = 0.1  # use last 10% of training for final-performance summary
BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(0)


@dataclass
class RunData:
    event_file: Path
    seed: int
    learning_rate: float
    steps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    returns: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


HPARAM_ROW = re.compile(r"^\|([^|]+)\|([^|]+)\|$")


def _parse_hparams(markdown: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in markdown.splitlines():
        m = HPARAM_ROW.match(line.strip())
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        if key in ("param", "-"):
            continue
        out[key] = value
    return out


def load_event_file(path: Path) -> RunData | None:
    """Parse a single TB event file. Returns None if it lacks the metric or hparams."""
    loader = EventFileLoader(str(path))
    hparams: dict[str, str] = {}
    steps: list[int] = []
    returns: list[float] = []
    for event in loader.Load():
        if not event.HasField("summary"):
            continue
        for v in event.summary.value:
            if v.tag == HPARAMS_TAG and v.HasField("tensor") and v.tensor.string_val:
                hparams = _parse_hparams(v.tensor.string_val[0].decode("utf-8", "ignore"))
            elif v.tag == METRIC:
                steps.append(event.step)
                # Scalars in TB v2 are stored as tensors with a single float value.
                if v.HasField("tensor") and len(v.tensor.float_val) > 0:
                    returns.append(float(v.tensor.float_val[0]))
                elif v.HasField("simple_value"):
                    returns.append(float(v.simple_value))
                else:
                    returns.append(float("nan"))
    if not hparams or not steps:
        return None
    try:
        seed = int(hparams["seed"])
        lr = float(hparams["learning_rate"])
    except (KeyError, ValueError):
        return None
    order = np.argsort(steps)
    return RunData(
        event_file=path,
        seed=seed,
        learning_rate=lr,
        steps=np.asarray(steps, dtype=np.int64)[order],
        returns=np.asarray(returns, dtype=np.float64)[order],
    )


def discover_runs(runs_dir: Path) -> list[RunData]:
    runs: list[RunData] = []
    for ef in sorted(runs_dir.rglob("events.out.tfevents.*")):
        rd = load_event_file(ef)
        if rd is not None:
            runs.append(rd)
    return runs


def dedupe_runs(runs: list[RunData]) -> list[RunData]:
    """If multiple event files share (lr, seed), keep the longest one."""
    best: dict[tuple[float, int], RunData] = {}
    for r in runs:
        key = (r.learning_rate, r.seed)
        prev = best.get(key)
        if prev is None or r.returns.size > prev.returns.size:
            best[key] = r
    return list(best.values())


def bin_returns(run: RunData, edges: np.ndarray) -> np.ndarray:
    """Mean episodic_return inside each bin defined by `edges` (NaN if empty)."""
    out = np.full(edges.size - 1, np.nan, dtype=np.float64)
    if run.steps.size == 0:
        return out
    idx = np.digitize(run.steps, edges) - 1  # bins indexed 0..N-1
    for b in range(edges.size - 1):
        mask = idx == b
        if mask.any():
            out[b] = run.returns[mask].mean()
    # Forward-fill empty bins so curves don't have visual gaps (rare with N=50).
    last = np.nan
    for i in range(out.size):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def bootstrap_mean_ci(
    samples: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a 1-D array of seed-level statistics."""
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return (float("nan"),) * 3
    if samples.size == 1:
        return float(samples[0]), float(samples[0]), float(samples[0])
    boots = RNG.choice(samples, size=(n_resamples, samples.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(samples.mean()), float(lo), float(hi)


def curve_ci(seed_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean, lo, hi) across seeds. `seed_curves` shape (n_seeds, n_bins)."""
    n_seeds, n_bins = seed_curves.shape
    mean = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        col = seed_curves[:, b]
        m, l, h = bootstrap_mean_ci(col)
        mean[b], lo[b], hi[b] = m, l, h
    return mean, lo, hi


def final_perf_per_seed(seed_curves: np.ndarray, frac: float = FINAL_FRAC) -> np.ndarray:
    n_bins = seed_curves.shape[1]
    k = max(1, int(round(frac * n_bins)))
    return np.nanmean(seed_curves[:, -k:], axis=1)


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pvals[i] * (n - rank))
        adj[i] = min(running, 1.0)
    return adj


def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--total-steps", type=int, default=100_000,
                        help="Right edge of the binning grid.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.runs_dir)
    print(f"Found {len(runs)} event files with the metric.")
    runs = dedupe_runs(runs)
    print(f"After de-dup by (lr, seed): {len(runs)} unique runs.")

    by_lr: dict[float, list[RunData]] = defaultdict(list)
    for r in runs:
        by_lr[r.learning_rate].append(r)
    lrs = sorted(by_lr)

    print("\nRuns per learning rate:")
    for lr in lrs:
        seeds = sorted(r.seed for r in by_lr[lr])
        print(f"  lr={fmt_lr(lr)}: {len(seeds)} seeds {seeds}")

    edges = np.linspace(0, args.total_steps, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Build (n_seeds, n_bins) curve per LR.
    binned: dict[float, np.ndarray] = {}
    for lr in lrs:
        rs = sorted(by_lr[lr], key=lambda r: r.seed)
        curves = np.stack([bin_returns(r, edges) for r in rs], axis=0)
        binned[lr] = curves

    # ---- Figure 1: learning curves with 95% bootstrap CIs ----
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lrs)))
    for lr, color in zip(lrs, colors):
        mean, lo, hi = curve_ci(binned[lr])
        ax.plot(centers, mean, label=f"lr = {fmt_lr(lr)} (n={binned[lr].shape[0]})",
                color=color, linewidth=2)
        ax.fill_between(centers, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.set_xlabel("environment step")
    ax.set_ylabel("episodic return (mean across seeds)")
    ax.set_title("CartPole-v1 / PPO — learning curves (95% bootstrap CI over seeds)")
    ax.axhline(500.0, color="grey", linestyle=":", linewidth=1, label="env max (500)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    curves_path = args.out_dir / "learning_curves.png"
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)
    print(f"\nWrote {curves_path}")

    # ---- Figure 2: per-seed curves (sanity / variance) ----
    fig, axes = plt.subplots(1, len(lrs), figsize=(5.2 * len(lrs), 4.2), sharey=True)
    if len(lrs) == 1:
        axes = [axes]
    for ax, lr, color in zip(axes, lrs, colors):
        curves = binned[lr]
        for i in range(curves.shape[0]):
            ax.plot(centers, curves[i], alpha=0.75, linewidth=1.2)
        ax.set_title(f"lr = {fmt_lr(lr)}")
        ax.set_xlabel("environment step")
        ax.grid(True, alpha=0.3)
        ax.axhline(500.0, color="grey", linestyle=":", linewidth=1)
    axes[0].set_ylabel("episodic return")
    fig.suptitle("Per-seed learning curves")
    fig.tight_layout()
    per_seed_path = args.out_dir / "per_seed_curves.png"
    fig.savefig(per_seed_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {per_seed_path}")

    # ---- Final-performance summary ----
    finals: dict[float, np.ndarray] = {lr: final_perf_per_seed(binned[lr]) for lr in lrs}

    print(f"\nFinal performance (mean over last {int(FINAL_FRAC * 100)}% of training):")
    print(f"{'lr':>10}  {'n':>3}  {'mean':>8}  {'std':>8}  {'sem':>8}  {'95% CI (bootstrap)':>26}")
    summary_rows = []
    for lr in lrs:
        v = finals[lr]
        m, lo, hi = bootstrap_mean_ci(v)
        std = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        sem = std / np.sqrt(v.size) if v.size > 0 else 0.0
        print(f"{fmt_lr(lr):>10}  {v.size:>3}  {m:>8.2f}  {std:>8.2f}  {sem:>8.2f}"
              f"  [{lo:>7.2f}, {hi:>7.2f}]")
        summary_rows.append((fmt_lr(lr), v.size, m, std, sem, lo, hi))

    # ---- Pairwise Welch's t-tests with Holm correction ----
    pairs = list(combinations(lrs, 2))
    raw_p, tstats, diffs = [], [], []
    for a, b in pairs:
        t = stats.ttest_ind(finals[a], finals[b], equal_var=False, nan_policy="omit")
        raw_p.append(float(t.pvalue))
        tstats.append(float(t.statistic))
        diffs.append(float(np.nanmean(finals[a]) - np.nanmean(finals[b])))
    adj_p = holm_correct(raw_p)

    print("\nPairwise Welch's t-tests on final performance (Holm-corrected):")
    print(f"{'lr_a':>10}  {'lr_b':>10}  {'mean_a - mean_b':>16}  {'t':>7}  {'p':>9}  {'p_adj':>9}")
    pair_rows = []
    for (a, b), t, d, p, pa in zip(pairs, tstats, diffs, raw_p, adj_p):
        print(f"{fmt_lr(a):>10}  {fmt_lr(b):>10}  {d:>16.2f}  {t:>7.2f}  {p:>9.4f}  {pa:>9.4f}")
        pair_rows.append((fmt_lr(a), fmt_lr(b), d, t, p, pa))

    # ---- Figure 3: final-performance bar chart with per-seed dots ----
    fig, ax = plt.subplots(figsize=(1.6 * len(lrs) + 2.0, 4.6))
    xs = np.arange(len(lrs))
    means = [bootstrap_mean_ci(finals[lr])[0] for lr in lrs]
    los = [bootstrap_mean_ci(finals[lr])[1] for lr in lrs]
    his = [bootstrap_mean_ci(finals[lr])[2] for lr in lrs]
    yerr = np.array([[m - lo for m, lo in zip(means, los)],
                     [hi - m for m, hi in zip(means, his)]])
    ax.bar(xs, means, yerr=yerr, capsize=6,
           color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    for x, lr in zip(xs, lrs):
        v = finals[lr]
        ax.scatter(np.full_like(v, x) + RNG.uniform(-0.08, 0.08, size=v.size),
                   v, color="black", s=22, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"lr = {fmt_lr(lr)}" for lr in lrs])
    ax.set_ylabel(f"mean episodic return (last {int(FINAL_FRAC * 100)}% of training)")
    ax.set_title("Final performance per learning rate (95% bootstrap CI, dots = seeds)")
    ax.axhline(500.0, color="grey", linestyle=":", linewidth=1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    final_path = args.out_dir / "final_performance.png"
    fig.savefig(final_path, dpi=160)
    plt.close(fig)
    print(f"\nWrote {final_path}")

    # ---- CSVs ----
    summary_csv = args.out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr", "n_seeds", "mean", "std", "sem", "ci95_lo", "ci95_hi"])
        w.writerows(summary_rows)

    pairs_csv = args.out_dir / "pairwise.csv"
    with pairs_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_a", "lr_b", "mean_a_minus_b", "t_stat", "p_raw", "p_holm"])
        w.writerows(pair_rows)

    per_seed_csv = args.out_dir / "per_seed_final.csv"
    with per_seed_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr", "seed", "final_mean_return"])
        for lr in lrs:
            for r, v in zip(sorted(by_lr[lr], key=lambda r: r.seed), finals[lr]):
                w.writerow([fmt_lr(lr), r.seed, f"{v:.4f}"])

    print(f"\nWrote {summary_csv}, {pairs_csv}, {per_seed_csv}")


if __name__ == "__main__":
    main()

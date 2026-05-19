# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11", "tensorboard>=2.18"]
# ///
"""Compare every LunarLander config tried so far.

Walks `./runs/` and groups runs by full hyperparameter config (lr, num_steps,
anneal_lr, total_timesteps, update_epochs). For each config it reports:

  * terminal mean return — mean over the last 10% of *its own* training,
    averaged across seeds (with 95% bootstrap CI). Note: configs differ in
    total_timesteps (100k vs 800k), so this metric rewards methods that were
    allowed to run longer.

  * early-budget mean return — mean over the last 10% of the *first 100k
    steps* (the smallest budget any method got). Fair across all methods.

  * AUC@100k — step-weighted mean return integrated over [0, 100k], averaged
    across seeds. Higher = faster learner under a fixed 100k budget.

  * time-to-return — first step at which the trailing-50-episode rolling mean
    crosses the threshold (default 100). Lower = faster. Configs where no
    seed crosses the threshold within 100k are flagged.

The script writes a ranked CSV, prints the top configs by terminal reward and
by speed, and produces overlay learning curves for the top configs of each
criterion.

Run from this directory: `uv run compare_all.py`.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _tblog import RunData, bin_returns, discover_all

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
OUT_DIR = HERE / "analysis_compare_all"

FAIR_BUDGET = 100_000  # smallest total_timesteps across all sweeps
N_BINS_FAIR = 50
FINAL_FRAC = 0.1
THRESHOLD = 100.0          # "good return" threshold for time-to-return
ROLLING_K = 50              # episodes to average for time-to-return
BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(0)


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


def terminal_return(run: RunData, frac: float = FINAL_FRAC) -> float:
    """Mean episodic return over the last `frac` of this run's training."""
    if run.steps.size == 0:
        return float("nan")
    cutoff = (1.0 - frac) * run.total_timesteps
    mask = run.steps >= cutoff
    if not mask.any():
        return float(run.returns[-max(1, run.returns.size // 10):].mean())
    return float(run.returns[mask].mean())


def return_at_budget(run: RunData, budget: int, window_frac: float = FINAL_FRAC) -> float:
    """Mean episodic return over the last `window_frac` of [0, budget]."""
    if run.steps.size == 0:
        return float("nan")
    lo = (1.0 - window_frac) * budget
    mask = (run.steps >= lo) & (run.steps <= budget)
    if not mask.any():
        return float("nan")
    return float(run.returns[mask].mean())


def auc_at_budget(run: RunData, budget: int) -> float:
    """Trapezoid mean return over [0, budget]. NaN if not enough points."""
    mask = run.steps <= budget
    s, r = run.steps[mask], run.returns[mask]
    if s.size < 2 or s[-1] - s[0] <= 0:
        return float("nan")
    return float(np.trapezoid(r, s) / (s[-1] - s[0]))


def time_to_threshold(run: RunData, threshold: float, budget: int,
                      k: int = ROLLING_K) -> float | None:
    """First step where the last-k rolling mean of returns ≥ threshold,
    restricted to steps ≤ budget. Returns None if never crossed."""
    mask = run.steps <= budget
    if not mask.any():
        return None
    s, r = run.steps[mask], run.returns[mask]
    if s.size < k:
        return None
    csum = np.cumsum(r)
    csum = np.concatenate(([0.0], csum))
    rolling = (csum[k:] - csum[:-k]) / k  # length s.size - k + 1
    rolling_steps = s[k - 1:]
    hits = np.where(rolling >= threshold)[0]
    if hits.size == 0:
        return None
    return float(rolling_steps[hits[0]])


@dataclass
class ConfigSummary:
    config_key: tuple
    label: str
    n_seeds: int
    runs: list[RunData]
    terminal_mean: float
    terminal_lo: float
    terminal_hi: float
    early_mean: float
    early_lo: float
    early_hi: float
    auc_mean: float
    auc_lo: float
    auc_hi: float
    time_to_thresh: list[float | None]  # per seed (None = never)

    @property
    def time_to_thresh_median(self) -> float:
        vals = [t for t in self.time_to_thresh if t is not None]
        if not vals:
            return float("inf")
        return float(np.median(vals))

    @property
    def time_to_thresh_reached(self) -> str:
        n = sum(t is not None for t in self.time_to_thresh)
        return f"{n}/{len(self.time_to_thresh)}"


def summarize_config(runs: list[RunData]) -> ConfigSummary:
    terminals = np.array([terminal_return(r) for r in runs])
    earlies = np.array([return_at_budget(r, FAIR_BUDGET) for r in runs])
    aucs = np.array([auc_at_budget(r, FAIR_BUDGET) for r in runs])
    times = [time_to_threshold(r, THRESHOLD, FAIR_BUDGET) for r in runs]
    t_m, t_lo, t_hi = bootstrap_mean_ci(terminals)
    e_m, e_lo, e_hi = bootstrap_mean_ci(earlies)
    a_m, a_lo, a_hi = bootstrap_mean_ci(aucs)
    budgets = sorted({r.total_timesteps for r in runs})
    budget_str = (f"{budgets[0] // 1000}k" if len(budgets) == 1
                  else f"{budgets[0] // 1000}–{budgets[-1] // 1000}k")
    label = f"{runs[0].config_label()}, T={budget_str}"
    return ConfigSummary(
        config_key=runs[0].config_key(),
        label=label,
        n_seeds=len(runs),
        runs=runs,
        terminal_mean=t_m, terminal_lo=t_lo, terminal_hi=t_hi,
        early_mean=e_m, early_lo=e_lo, early_hi=e_hi,
        auc_mean=a_m, auc_lo=a_lo, auc_hi=a_hi,
        time_to_thresh=times,
    )


def overlay_curves(summaries: list[ConfigSummary], budget: int | None,
                   title: str, path: Path) -> None:
    """If budget is None, each method is plotted up to its own min-seed budget."""
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(summaries), 10)))
    for s, color in zip(summaries, colors):
        # Per-method range: use the explicit budget if given, else the smallest
        # seed budget so we never forward-fill past observed data.
        method_budget = budget if budget is not None else min(r.total_timesteps for r in s.runs)
        edges = np.linspace(0, method_budget, N_BINS_FAIR + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        curves = np.stack([bin_returns(r, edges) for r in s.runs], axis=0)
        n_bins = curves.shape[1]
        mean = np.full(n_bins, np.nan)
        lo = np.full(n_bins, np.nan)
        hi = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m, l, h = bootstrap_mean_ci(curves[:, b])
            mean[b], lo[b], hi[b] = m, l, h
        ax.plot(centers, mean, color=color, linewidth=2,
                label=f"{s.label}  (n={s.n_seeds})")
        ax.fill_between(centers, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.axhline(THRESHOLD, color="grey", linestyle=":", linewidth=1,
               label=f"return = {int(THRESHOLD)}")
    ax.axhline(200.0, color="grey", linestyle="--", linewidth=1, label="solved (200)")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("environment step")
    ax.set_ylabel("episodic return (mean across seeds)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def speed_vs_quality_scatter(summaries: list[ConfigSummary], path: Path) -> None:
    """AUC@100k (x) vs terminal mean return (y), one dot per config."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs = np.array([s.auc_mean for s in summaries])
    ys = np.array([s.terminal_mean for s in summaries])
    ax.errorbar(
        xs, ys,
        xerr=[xs - np.array([s.auc_lo for s in summaries]),
              np.array([s.auc_hi for s in summaries]) - xs],
        yerr=[ys - np.array([s.terminal_lo for s in summaries]),
              np.array([s.terminal_hi for s in summaries]) - ys],
        fmt="o", capsize=3, markersize=6, linewidth=0.8, alpha=0.85, color="C0",
    )
    for s, x, y in zip(summaries, xs, ys):
        ax.annotate(s.label, xy=(x, y), xytext=(5, 4), textcoords="offset points",
                    fontsize=7, alpha=0.85)
    ax.axhline(200.0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("AUC@100k  (higher = faster learner over first 100k)")
    ax.set_ylabel("terminal mean return  (mean over last 10% of own training)")
    ax.set_title("Speed (AUC@100k) vs. quality (terminal return) per config")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top configs to overlay in summary plots.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Dedup by (algorithmic config, seed): prefer the run with the longest
    # training budget, then the one with the longest series.
    all_runs = discover_all(args.runs_dir)
    best: dict[tuple, RunData] = {}
    for r in all_runs:
        key = (r.config_key(), r.seed)
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        if (r.total_timesteps, r.returns.size) > (prev.total_timesteps, prev.returns.size):
            best[key] = r
    runs = list(best.values())
    print(f"Loaded {len(runs)} unique (method, seed) runs.")

    by_cfg: dict[tuple, list[RunData]] = defaultdict(list)
    for r in runs:
        by_cfg[r.config_key()].append(r)

    summaries = [summarize_config(rs) for rs in by_cfg.values()]
    print(f"Distinct configs: {len(summaries)}")

    # ---- Master table ----
    sort_terminal = sorted(summaries, key=lambda s: -s.terminal_mean)
    sort_speed = sorted(summaries, key=lambda s: -s.auc_mean)
    sort_time = sorted(summaries,
                       key=lambda s: (s.time_to_thresh_median, -s.early_mean))

    csv_path = args.out_dir / "all_configs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "lr", "num_steps", "anneal_lr", "update_epochs", "n_seeds",
            "budget_min", "budget_max",
            "terminal_mean", "terminal_ci_lo", "terminal_ci_hi",
            "early_at_100k_mean", "early_at_100k_ci_lo", "early_at_100k_ci_hi",
            "auc_100k_mean", "auc_100k_ci_lo", "auc_100k_ci_hi",
            "time_to_R100_median_step", "time_to_R100_reached",
        ])
        for s in sort_terminal:
            lr, ns, anneal, ue = s.config_key
            tmed = s.time_to_thresh_median
            budgets = sorted({r.total_timesteps for r in s.runs})
            w.writerow([
                lr, ns, anneal, ue, s.n_seeds,
                budgets[0], budgets[-1],
                f"{s.terminal_mean:.2f}", f"{s.terminal_lo:.2f}", f"{s.terminal_hi:.2f}",
                f"{s.early_mean:.2f}", f"{s.early_lo:.2f}", f"{s.early_hi:.2f}",
                f"{s.auc_mean:.2f}", f"{s.auc_lo:.2f}", f"{s.auc_hi:.2f}",
                "" if not np.isfinite(tmed) else f"{tmed:.0f}",
                s.time_to_thresh_reached,
            ])
    print(f"\nWrote {csv_path}")

    def print_table(title: str, ordered: list[ConfigSummary], cols: list[str]) -> None:
        print(f"\n{title}")
        header = f"{'#':>2}  " + "  ".join(f"{c:>22}" for c in cols)
        print(header)
        print("-" * len(header))
        for rank, s in enumerate(ordered[: args.top_k], start=1):
            row = {
                "config": s.label,
                "n_seeds": str(s.n_seeds),
                "terminal": f"{s.terminal_mean:>7.1f}  [{s.terminal_lo:>6.1f},{s.terminal_hi:>6.1f}]",
                "early@100k": f"{s.early_mean:>7.1f}  [{s.early_lo:>6.1f},{s.early_hi:>6.1f}]",
                "AUC@100k": f"{s.auc_mean:>7.1f}  [{s.auc_lo:>6.1f},{s.auc_hi:>6.1f}]",
                "time_to_R100": (
                    f"{'-' if not np.isfinite(s.time_to_thresh_median) else f'{s.time_to_thresh_median:.0f}'}"
                    f"  ({s.time_to_thresh_reached})"
                ),
            }
            print(f"{rank:>2}  " + "  ".join(f"{row[c]:>22}" for c in cols))

    print_table(
        f"Top {args.top_k} by TERMINAL MEAN RETURN (mean over last 10% of each run's own training):",
        sort_terminal,
        ["config", "n_seeds", "terminal", "early@100k", "AUC@100k"],
    )
    print_table(
        f"Top {args.top_k} by AUC@100k (fastest learners over the first 100k steps):",
        sort_speed,
        ["config", "n_seeds", "AUC@100k", "early@100k", "terminal"],
    )
    print_table(
        f"Top {args.top_k} by FASTEST TIME-TO-RETURN-{int(THRESHOLD)} within 100k steps:",
        sort_time,
        ["config", "n_seeds", "time_to_R100", "AUC@100k", "terminal"],
    )

    # ---- Overlay plots ----
    top_quality = sort_terminal[: args.top_k]
    top_speed = sort_speed[: args.top_k]

    overlay_curves(
        top_quality, budget=FAIR_BUDGET,
        title=f"Top {args.top_k} methods by terminal return — first {FAIR_BUDGET // 1000}k steps",
        path=args.out_dir / "top_terminal_curves_100k.png",
    )
    overlay_curves(
        top_quality, budget=None,
        title=f"Top {args.top_k} methods by terminal return — full training (each method up to its min seed budget)",
        path=args.out_dir / "top_terminal_curves_full.png",
    )
    overlay_curves(
        top_speed, budget=FAIR_BUDGET,
        title=f"Top {args.top_k} methods by AUC@100k — first {FAIR_BUDGET // 1000}k steps",
        path=args.out_dir / "top_speed_curves_100k.png",
    )

    speed_vs_quality_scatter(summaries, args.out_dir / "speed_vs_quality.png")
    print(f"\nWrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()

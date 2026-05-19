# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26", "tensorboard>=2.18"]
# ///
"""Compare the LunarLander PPO num_steps sweep.

The sweep runs one seed per `num_steps` value at fixed lr=0.003 (see
`lander_ppo.sh`), so with n=1 we can't do across-seed inference. Instead this
script reports *within-run* statistics: rolling-mean learning curves with a
shaded SE band computed from episode-to-episode variation inside each bin, and
final-performance summaries (mean ± SE of episodic return over the last 10% of
training).

Run from this directory: `uv run analyze_nsteps.py`.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
OUT_DIR = HERE / "analysis_nsteps"

METRIC = "charts/episodic_return"
HPARAMS_TAG = "hyperparameters/text_summary"
N_BINS = 50
FINAL_FRAC = 0.1


@dataclass
class RunData:
    event_file: Path
    seed: int
    learning_rate: float
    num_steps: int
    exp_name: str
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
        num_steps = int(hparams["num_steps"])
        exp_name = hparams.get("exp_name", "")
    except (KeyError, ValueError):
        return None
    order = np.argsort(steps)
    return RunData(
        event_file=path,
        seed=seed,
        learning_rate=lr,
        num_steps=num_steps,
        exp_name=exp_name,
        steps=np.asarray(steps, dtype=np.int64)[order],
        returns=np.asarray(returns, dtype=np.float64)[order],
    )


def discover_sweep_runs(runs_dir: Path) -> list[RunData]:
    """Pick up only the num_steps sweep — runs whose exp_name contains 'nsteps'."""
    out: list[RunData] = []
    for ef in sorted(runs_dir.rglob("events.out.tfevents.*")):
        # Cheap path filter first so we don't parse unrelated runs.
        if "nsteps" not in ef.parent.name:
            continue
        rd = load_event_file(ef)
        if rd is None or "nsteps" not in rd.exp_name:
            continue
        out.append(rd)
    return out


def dedupe(runs: list[RunData]) -> list[RunData]:
    """If multiple event files share (num_steps, seed), keep the longest one."""
    best: dict[tuple[int, int], RunData] = {}
    for r in runs:
        key = (r.num_steps, r.seed)
        prev = best.get(key)
        if prev is None or r.returns.size > prev.returns.size:
            best[key] = r
    return sorted(best.values(), key=lambda r: r.num_steps)


def bin_stats(
    run: RunData, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean, sem, count) of episodic returns, NaN where empty."""
    n_bins = edges.size - 1
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    if run.steps.size == 0:
        return means, sems, counts
    idx = np.digitize(run.steps, edges) - 1
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        counts[b] = c
        if c == 0:
            continue
        vals = run.returns[mask]
        means[b] = vals.mean()
        sems[b] = vals.std(ddof=1) / np.sqrt(c) if c > 1 else 0.0
    return means, sems, counts


def forward_fill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    last = np.nan
    for i in range(out.size):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def final_window_stats(run: RunData, total_steps: int, frac: float = FINAL_FRAC) -> tuple[float, float, int]:
    """(mean, sem, n) of returns whose step >= (1 - frac) * total_steps."""
    cutoff = (1.0 - frac) * total_steps
    mask = run.steps >= cutoff
    vals = run.returns[mask]
    n = vals.size
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def auc(run: RunData, total_steps: int) -> float:
    """Step-weighted mean return over [0, total_steps] via trapezoid rule.

    Truncates points beyond total_steps; returns NaN if too few points."""
    s, r = run.steps, run.returns
    mask = s <= total_steps
    s, r = s[mask], r[mask]
    if s.size < 2:
        return float("nan")
    return float(np.trapezoid(r, s) / (s[-1] - s[0]))


def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--total-steps", type=int, default=100_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = dedupe(discover_sweep_runs(args.runs_dir))
    if not runs:
        raise SystemExit(f"No nsteps-sweep runs found under {args.runs_dir}.")

    # Sanity: expect a single (lr, seed) configuration; flag anything unexpected.
    lrs = {r.learning_rate for r in runs}
    seeds = {r.seed for r in runs}
    print(f"Loaded {len(runs)} runs from the num_steps sweep.")
    print(f"  learning_rate: {sorted(lrs)}")
    print(f"  seeds:         {sorted(seeds)}")
    if len(seeds) == 1:
        print("  NOTE: only 1 seed per config — no across-seed inference is possible;")
        print("        SE bands below come from episode-to-episode variation inside each run.")

    edges = np.linspace(0, args.total_steps, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))

    # ---- Figure 1: smoothed learning curves with within-bin SE bands ----
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for r, color in zip(runs, colors):
        means, sems, _ = bin_stats(r, edges)
        m_filled = forward_fill(means)
        se_filled = forward_fill(sems)
        ax.plot(centers, m_filled, color=color, linewidth=2,
                label=f"num_steps = {r.num_steps}")
        ax.fill_between(centers, m_filled - se_filled, m_filled + se_filled,
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("environment step")
    ax.set_ylabel("episodic return (per-bin mean ± SE)")
    ax.set_title(
        f"LunarLander-v3 / PPO — num_steps sweep "
        f"(lr={fmt_lr(next(iter(lrs)))}, seed={sorted(seeds)[0]}, no LR anneal)"
    )
    ax.axhline(200.0, color="grey", linestyle=":", linewidth=1, label="solved (200)")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    curves_path = args.out_dir / "learning_curves.png"
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)
    print(f"\nWrote {curves_path}")

    # ---- Figure 2: small multiples with raw episodes underlaid ----
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r, color in zip(axes, runs, colors):
        means, _, _ = bin_stats(r, edges)
        ax.scatter(r.steps, r.returns, s=6, color=color, alpha=0.25, linewidths=0,
                   label="episode")
        ax.plot(centers, forward_fill(means), color=color, linewidth=2,
                label="binned mean")
        ax.axhline(200.0, color="grey", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.set_xlabel("environment step")
        ax.set_title(f"num_steps = {r.num_steps}  (n_eps = {r.steps.size})")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("episodic return")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-run learning curves")
    fig.tight_layout()
    raw_path = args.out_dir / "per_run_curves.png"
    fig.savefig(raw_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {raw_path}")

    # ---- Final-performance summary + AUC ----
    print(f"\nFinal {int(FINAL_FRAC * 100)}% of training (episode-level mean ± SE):")
    print(f"{'num_steps':>10}  {'n_eps':>6}  {'final_mean':>11}  {'final_sem':>10}"
          f"  {'auc':>10}  {'best_ep':>8}")
    rows = []
    for r in runs:
        fmean, fsem, nfinal = final_window_stats(r, args.total_steps)
        a = auc(r, args.total_steps)
        best = float(np.max(r.returns)) if r.returns.size else float("nan")
        print(f"{r.num_steps:>10}  {nfinal:>6}  {fmean:>11.2f}  {fsem:>10.2f}"
              f"  {a:>10.2f}  {best:>8.2f}")
        rows.append((r.num_steps, r.steps.size, nfinal, fmean, fsem, a, best))

    # ---- Figure 3: final performance bar chart (within-run SE error bars) ----
    fig, ax = plt.subplots(figsize=(1.6 * len(runs) + 2.0, 4.6))
    xs = np.arange(len(runs))
    finals = [final_window_stats(r, args.total_steps) for r in runs]
    means = [f[0] for f in finals]
    sems = [f[1] for f in finals]
    ax.bar(xs, means, yerr=sems, capsize=6,
           color=colors, alpha=0.75, edgecolor="black", linewidth=0.5)
    for x, r in zip(xs, runs):
        cutoff = (1.0 - FINAL_FRAC) * args.total_steps
        vals = r.returns[r.steps >= cutoff]
        ax.scatter(np.full_like(vals, x, dtype=float)
                   + np.random.default_rng(0).uniform(-0.10, 0.10, size=vals.size),
                   vals, color="black", s=8, alpha=0.45, linewidths=0, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ns={r.num_steps}" for r in runs])
    ax.set_ylabel(f"episodic return — last {int(FINAL_FRAC * 100)}% (mean ± SE)")
    ax.set_title("Final-window return per num_steps (dots = individual episodes)")
    ax.axhline(200.0, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    final_path = args.out_dir / "final_performance.png"
    fig.savefig(final_path, dpi=160)
    plt.close(fig)
    print(f"\nWrote {final_path}")

    # ---- CSV ----
    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_steps", "n_episodes", "n_final_window", "final_mean",
                    "final_sem", "auc", "best_episode"])
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

"""Aggregate CartPole-v1 runs in runs/ and produce a comparison report.

Reads TensorBoard event files under runs/CartPole-v1__{algo}__{seed}__{ts}/,
extracts charts/episodic_return, and emits:
  - scripts/cartpole_results.png (mean +/- std learning curves per algorithm)
  - scripts/cartpole_summary.md  (table of metrics per algorithm)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"
OUT_PNG = REPO / "scripts" / "cartpole_results.png"
OUT_MD = REPO / "scripts" / "cartpole_summary.md"

ALGOS = ["ppo", "dqn", "c51", "pqn"]
SEEDS = [1, 2, 3, 4, 5]
SOLVED_THRESHOLD = 475.0
FINAL_WINDOW_STEPS = 25_000
TOTAL_TIMESTEPS = 500_000
BIN_WIDTH = 5_000  # for step-binning learning curves across seeds

RUN_RE = re.compile(r"^CartPole-v1__([a-z0-9]+)__(\d+)__\d+$")


@dataclass
class Run:
    algo: str
    seed: int
    path: Path
    steps: np.ndarray
    returns: np.ndarray
    wall_times: np.ndarray
    test_return: float = float("nan")

    @property
    def duration_seconds(self) -> float:
        if len(self.wall_times) < 2:
            return float("nan")
        return float(self.wall_times[-1] - self.wall_times[0])

    def first_solved_step(self) -> float:
        hit = self.steps[self.returns >= SOLVED_THRESHOLD]
        return float(hit[0]) if len(hit) else float("nan")

    def final_window_mean(self) -> float:
        mask = self.steps >= (TOTAL_TIMESTEPS - FINAL_WINDOW_STEPS)
        return float(self.returns[mask].mean()) if mask.any() else float("nan")

    def final_window_std(self) -> float:
        mask = self.steps >= (TOTAL_TIMESTEPS - FINAL_WINDOW_STEPS)
        return float(self.returns[mask].std()) if mask.any() else float("nan")


def load_run(path: Path) -> Run | None:
    m = RUN_RE.match(path.name)
    if not m:
        return None
    algo, seed = m.group(1), int(m.group(2))
    if algo not in ALGOS or seed not in SEEDS:
        return None
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "charts/episodic_return" not in tags:
        return None
    events = ea.Scalars("charts/episodic_return")
    if not events:
        return None
    steps = np.array([e.step for e in events], dtype=np.float64)
    returns = np.array([e.value for e in events], dtype=np.float64)
    wall_times = np.array([e.wall_time for e in events], dtype=np.float64)
    test_return = float("nan")
    if "eval/test_episodic_return" in tags:
        test_events = ea.Scalars("eval/test_episodic_return")
        if test_events:
            test_return = float(test_events[-1].value)
    return Run(
        algo=algo, seed=seed, path=path,
        steps=steps, returns=returns, wall_times=wall_times, test_return=test_return,
    )


def bin_to_grid(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Bin (steps, values) into fixed grid by averaging points in each bin.
    Forward-fills empty bins with the last known value (and back-fills any leading gap)."""
    out = np.full(len(grid), np.nan, dtype=np.float64)
    bin_idx = np.searchsorted(grid, steps, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(grid) - 1)
    for i in range(len(grid)):
        sel = values[bin_idx == i]
        if len(sel) > 0:
            out[i] = sel.mean()
    # fill NaNs: forward-fill, then back-fill
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    # back-fill leading NaNs
    first_valid = next((i for i, v in enumerate(out) if not np.isnan(v)), None)
    if first_valid is not None:
        out[:first_valid] = out[first_valid]
    return out


def plot_curves(grouped: dict[str, list[Run]]) -> None:
    grid = np.arange(0, TOTAL_TIMESTEPS + BIN_WIDTH, BIN_WIDTH, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"ppo": "#1f77b4", "dqn": "#ff7f0e", "c51": "#2ca02c", "pqn": "#d62728"}
    for algo in ALGOS:
        runs = grouped.get(algo, [])
        if not runs:
            continue
        curves = np.stack([bin_to_grid(r.steps, r.returns, grid) for r in runs])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(grid, mean, label=f"{algo.upper()} (n={len(runs)})", color=colors[algo], linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=colors[algo], alpha=0.2)
    ax.axhline(SOLVED_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"solved ({SOLVED_THRESHOLD:.0f})")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Episodic return")
    ax.set_title("CartPole-v1: learning curves (mean +/- std across seeds)")
    ax.set_xlim(0, TOTAL_TIMESTEPS)
    ax.set_ylim(0, 520)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")


def write_summary(grouped: dict[str, list[Run]]) -> None:
    lines: list[str] = []
    lines.append("# CartPole-v1 Algorithm Comparison\n")
    lines.append(f"- Total timesteps per run: {TOTAL_TIMESTEPS:,}")
    lines.append(f"- Seeds per algorithm: {len(SEEDS)} ({SEEDS})")
    lines.append(f"- \"Solved\" threshold: episodic return >= {SOLVED_THRESHOLD:.0f}")
    lines.append(f"- Final-window mean: average episodic return over last {FINAL_WINDOW_STEPS:,} steps\n")
    lines.append("## Aggregated metrics\n")
    lines.append("| Algorithm | Final return (mean +/- std) | Deterministic test return (mean +/- std) | Steps to solve (median) | Solved seeds | Wall-clock (s, mean) | Late-training std (mean) |")
    lines.append("|---|---|---|---|---|---|---|")
    for algo in ALGOS:
        runs = grouped.get(algo, [])
        if not runs:
            lines.append(f"| {algo.upper()} | no runs | - | - | - | - | - |")
            continue
        finals = np.array([r.final_window_mean() for r in runs])
        tests = np.array([r.test_return for r in runs])
        tests_valid = tests[~np.isnan(tests)]
        test_str = f"{tests_valid.mean():.1f} +/- {tests_valid.std():.1f}" if len(tests_valid) else "n/a"
        solves = np.array([r.first_solved_step() for r in runs])
        solved_mask = ~np.isnan(solves)
        med_solve = f"{int(np.median(solves[solved_mask])):,}" if solved_mask.any() else "never"
        durations = np.array([r.duration_seconds for r in runs])
        late_stds = np.array([r.final_window_std() for r in runs])
        lines.append(
            f"| {algo.upper()} "
            f"| {finals.mean():.1f} +/- {finals.std():.1f} "
            f"| {test_str} "
            f"| {med_solve} "
            f"| {int(solved_mask.sum())}/{len(runs)} "
            f"| {durations.mean():.1f} "
            f"| {late_stds.mean():.1f} |"
        )
    lines.append("\n## Per-run detail\n")
    lines.append("| Algorithm | Seed | Final return | Test return | First solved step | Duration (s) |")
    lines.append("|---|---|---|---|---|---|")
    for algo in ALGOS:
        for run in sorted(grouped.get(algo, []), key=lambda r: r.seed):
            solve = run.first_solved_step()
            solve_str = f"{int(solve):,}" if not np.isnan(solve) else "never"
            test_str = f"{run.test_return:.1f}" if not np.isnan(run.test_return) else "n/a"
            lines.append(
                f"| {algo.upper()} | {run.seed} | {run.final_window_mean():.1f} "
                f"| {test_str} | {solve_str} | {run.duration_seconds:.1f} |"
            )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


def main() -> None:
    if not RUNS_DIR.exists():
        raise SystemExit(f"runs dir not found: {RUNS_DIR}")
    grouped: dict[str, list[Run]] = {a: [] for a in ALGOS}
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        run = load_run(entry)
        if run is None:
            continue
        grouped[run.algo].append(run)
    total = sum(len(v) for v in grouped.values())
    print(f"Loaded {total} runs: " + ", ".join(f"{a}={len(grouped[a])}" for a in ALGOS))
    plot_curves(grouped)
    write_summary(grouped)


if __name__ == "__main__":
    main()

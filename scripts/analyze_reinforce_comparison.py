"""Aggregate multi-env runs in runs/ and produce a comparison report.

Groups by (env, algo), plots mean +/- std learning curves per env, and emits:
  - scripts/reinforce_results.png  (one subplot per env)
  - scripts/reinforce_summary.md   (per-env metric tables)
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
OUT_PNG = REPO / "scripts" / "reinforce_results.png"
OUT_MD = REPO / "scripts" / "reinforce_summary.md"

ALGOS = ["ppo", "reinforce", "reinforce_with_baseline"]
SEEDS = [1, 2, 3, 4, 5]
TOTAL_TIMESTEPS = 500_000
FINAL_WINDOW_STEPS = 50_000
BIN_WIDTH = 5_000

# (env_id, solved threshold, display-name, y-axis bottom)
ENVS: list[tuple[str, float, str, float]] = [
    ("CartPole-v1", 475.0,  "CartPole-v1", 0.0),
    ("Acrobot-v1",  -100.0, "Acrobot-v1",  -500.0),
]
ENV_IDS = [e[0] for e in ENVS]

# dirname: {env}__{algo}__{seed}__{unix_ts}
# env can contain hyphens (CartPole-v1, LunarLander-v3). Use non-greedy match.
RUN_RE = re.compile(r"^(.+?)__(.+?)__(\d+)__\d+$")

COLOR = {"ppo": "#1f77b4", "reinforce": "#d62728", "reinforce_with_baseline": "#2ca02c"}
LABEL = {"ppo": "PPO", "reinforce": "REINFORCE", "reinforce_with_baseline": "REINFORCE + baseline"}


@dataclass
class Run:
    env: str
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

    def first_threshold_step(self, threshold: float) -> float:
        hit = self.steps[self.returns >= threshold]
        return float(hit[0]) if len(hit) else float("nan")

    def final_window_mean(self) -> float:
        mask = self.steps >= (TOTAL_TIMESTEPS - FINAL_WINDOW_STEPS)
        return float(self.returns[mask].mean()) if mask.any() else float("nan")


def load_run(path: Path) -> Run | None:
    m = RUN_RE.match(path.name)
    if not m:
        return None
    env, algo, seed = m.group(1), m.group(2), int(m.group(3))
    if env not in ENV_IDS or algo not in ALGOS or seed not in SEEDS:
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
        te = ea.Scalars("eval/test_episodic_return")
        if te:
            test_return = float(te[-1].value)
    return Run(env=env, algo=algo, seed=seed, path=path,
               steps=steps, returns=returns, wall_times=wall_times, test_return=test_return)


def bin_to_grid(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.full(len(grid), np.nan, dtype=np.float64)
    bin_idx = np.searchsorted(grid, steps, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(grid) - 1)
    for i in range(len(grid)):
        sel = values[bin_idx == i]
        if len(sel) > 0:
            out[i] = sel.mean()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    first_valid = next((i for i, v in enumerate(out) if not np.isnan(v)), None)
    if first_valid is not None:
        out[:first_valid] = out[first_valid]
    return out


def plot_curves(grouped: dict[tuple[str, str], list[Run]]) -> None:
    grid = np.arange(0, TOTAL_TIMESTEPS + BIN_WIDTH, BIN_WIDTH, dtype=np.float64)
    fig, axes = plt.subplots(1, len(ENVS), figsize=(6 * len(ENVS), 5), sharex=True)
    if len(ENVS) == 1:
        axes = [axes]
    for ax, (env, solved, display, y_bottom) in zip(axes, ENVS):
        has_data = False
        for algo in ALGOS:
            runs = grouped.get((env, algo), [])
            if not runs:
                continue
            has_data = True
            curves = np.stack([bin_to_grid(r.steps, r.returns, grid) for r in runs])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            ax.plot(grid, mean, label=f"{LABEL[algo]} (n={len(runs)})", color=COLOR[algo], linewidth=2)
            ax.fill_between(grid, mean - std, mean + std, color=COLOR[algo], alpha=0.2)
        ax.axhline(solved, color="gray", linestyle="--", linewidth=1, label=f"solved ({solved:.0f})")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Episodic return")
        ax.set_title(display)
        ax.set_xlim(0, TOTAL_TIMESTEPS)
        ax.set_ylim(bottom=y_bottom)
        if has_data:
            ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("PPO vs REINFORCE vs REINFORCE+baseline (mean +/- std across 5 seeds)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")


def write_summary(grouped: dict[tuple[str, str], list[Run]]) -> None:
    lines: list[str] = []
    lines.append("# PPO vs REINFORCE vs REINFORCE+baseline\n")
    lines.append(f"- Total env-step budget per run: {TOTAL_TIMESTEPS:,}")
    lines.append(f"- Seeds per (env, algo) cell: {len(SEEDS)} ({SEEDS})")
    lines.append(f"- Final-window mean: average episodic return over last {FINAL_WINDOW_STEPS:,} steps\n")

    for env, solved, display, _ in ENVS:
        lines.append(f"\n## {display} (solved = {solved:g})\n")
        lines.append("| Algo | Final (mean +/- std) | Deterministic test | Steps to solve (median) | Solved | Wall-clock (s, mean) |")
        lines.append("|---|---|---|---|---|---|")
        for algo in ALGOS:
            runs = grouped.get((env, algo), [])
            if not runs:
                lines.append(f"| {LABEL[algo]} | no runs | - | - | - | - |")
                continue
            finals = np.array([r.final_window_mean() for r in runs])
            tests = np.array([r.test_return for r in runs])
            tests_valid = tests[~np.isnan(tests)]
            test_str = f"{tests_valid.mean():.1f} +/- {tests_valid.std():.1f}" if len(tests_valid) else "n/a"
            solves = np.array([r.first_threshold_step(solved) for r in runs])
            solved_mask = ~np.isnan(solves)
            med_solve = f"{int(np.median(solves[solved_mask])):,}" if solved_mask.any() else "never"
            durations = np.array([r.duration_seconds for r in runs])
            lines.append(
                f"| {LABEL[algo]} "
                f"| {finals.mean():.1f} +/- {finals.std():.1f} "
                f"| {test_str} "
                f"| {med_solve} "
                f"| {int(solved_mask.sum())}/{len(runs)} "
                f"| {durations.mean():.1f} |"
            )
        lines.append("\n### Per-seed detail\n")
        lines.append("| Algo | Seed | Final | Test | First solved | Duration (s) |")
        lines.append("|---|---|---|---|---|---|")
        for algo in ALGOS:
            for r in sorted(grouped.get((env, algo), []), key=lambda r: r.seed):
                s = r.first_threshold_step(solved)
                s_str = f"{int(s):,}" if not np.isnan(s) else "never"
                t_str = f"{r.test_return:.1f}" if not np.isnan(r.test_return) else "n/a"
                lines.append(
                    f"| {LABEL[algo]} | {r.seed} | {r.final_window_mean():.1f} "
                    f"| {t_str} | {s_str} | {r.duration_seconds:.1f} |"
                )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")


def main() -> None:
    if not RUNS_DIR.exists():
        raise SystemExit(f"runs dir not found: {RUNS_DIR}")
    grouped: dict[tuple[str, str], list[Run]] = {}
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        run = load_run(entry)
        if run is None:
            continue
        grouped.setdefault((run.env, run.algo), []).append(run)
    total = sum(len(v) for v in grouped.values())
    breakdown = ", ".join(
        f"{env}/{algo}={len(grouped.get((env, algo), []))}"
        for env in ENV_IDS for algo in ALGOS
    )
    print(f"Loaded {total} runs: {breakdown}")
    plot_curves(grouped)
    write_summary(grouped)


if __name__ == "__main__":
    main()

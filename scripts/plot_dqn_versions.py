"""Plot learning curves for stock DQN, dqn_alt v1 (before fix), dqn_alt v2 (after fix).

All curves truncated at 350k global steps for apples-to-apples comparison.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"
OUT_PNG = REPO / "scripts" / "dqn_versions_learning_curves.png"

MAX_STEP = 350_000
BIN_WIDTH = 5_000
SOLVED = 475.0

# (exp_name, label, color, allowed_seeds)
SERIES = [
    ("dqn",         "DQN (stock CleanRL)",                 "#ff7f0e", {1, 2, 3, 4, 5}),
    ("dqn_alt_v1",  "dqn_alt (before fixes)",              "#d62728", {1}),
    ("dqn_alt",     "dqn_alt (learning_starts + eps fix)", "#17becf", {1, 2}),
    ("dqn_alt_v3",  "dqn_alt (+ exploration_fraction=0.5)", "#2ca02c", {1, 2}),
    ("dqn_alt_v4",  "dqn_alt (+ train_frequency=10, lr=2.5e-4)", "#9467bd", {1, 2}),
]

RUN_RE = re.compile(r"^(.+?)__(.+?)__(\d+)__\d+$")


def load_runs_for(exp_name: str, allowed_seeds: set[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        m = RUN_RE.match(entry.name)
        if not m or m.group(1) != "CartPole-v1":
            continue
        if m.group(2) != exp_name:
            continue
        seed = int(m.group(3))
        if seed not in allowed_seeds:
            continue
        ea = EventAccumulator(str(entry), size_guidance={"scalars": 0})
        ea.Reload()
        if "charts/episodic_return" not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars("charts/episodic_return")
        if not events:
            continue
        steps = np.array([e.step for e in events], dtype=np.float64)
        returns = np.array([e.value for e in events], dtype=np.float64)
        mask = steps <= MAX_STEP
        out.append((steps[mask], returns[mask]))
    return out


def bin_to_grid(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.full(len(grid), np.nan, dtype=np.float64)
    bin_idx = np.clip(np.searchsorted(grid, steps, side="right") - 1, 0, len(grid) - 1)
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


def main() -> None:
    grid = np.arange(0, MAX_STEP + BIN_WIDTH, BIN_WIDTH, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_name, label, color, seeds in SERIES:
        runs = load_runs_for(exp_name, seeds)
        if not runs:
            print(f"WARNING: no runs found for {exp_name}")
            continue
        curves = np.stack([bin_to_grid(s, r, grid) for s, r in runs])
        mean = curves.mean(axis=0)
        ax.plot(grid, mean, label=f"{label} (n={len(runs)})", color=color, linewidth=2)
        if len(runs) > 1:
            std = curves.std(axis=0)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
        print(f"{exp_name}: n={len(runs)}, final value={mean[-1]:.1f}")
    ax.axhline(SOLVED, color="gray", linestyle="--", linewidth=1, label=f"solved ({SOLVED:.0f})")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Episodic return")
    ax.set_title("CartPole-v1: DQN vs dqn_alt (before/after fixes)")
    ax.set_xlim(0, MAX_STEP)
    ax.set_ylim(0, 520)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

"""Shared TensorBoard log loader for LunarLander analysis scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

METRIC = "charts/episodic_return"
HPARAMS_TAG = "hyperparameters/text_summary"

_HPARAM_ROW = re.compile(r"^\|([^|]+)\|([^|]+)\|$")


@dataclass
class RunData:
    event_file: Path
    hparams: dict[str, str]
    steps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    returns: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    @property
    def seed(self) -> int:
        return int(self.hparams["seed"])

    @property
    def learning_rate(self) -> float:
        return float(self.hparams["learning_rate"])

    @property
    def num_steps(self) -> int:
        return int(self.hparams["num_steps"])

    @property
    def anneal_lr(self) -> bool:
        return self.hparams.get("anneal_lr", "False") == "True"

    @property
    def total_timesteps(self) -> int:
        return int(self.hparams["total_timesteps"])

    @property
    def update_epochs(self) -> int:
        return int(self.hparams.get("update_epochs", "4"))

    @property
    def exp_name(self) -> str:
        return self.hparams.get("exp_name", "")

    def config_key(self) -> tuple:
        """Hashable key for the algorithmic config (excludes seed and budget)."""
        return (
            self.learning_rate,
            self.num_steps,
            self.anneal_lr,
            self.update_epochs,
        )

    def config_label(self) -> str:
        return (
            f"lr={self.learning_rate:g}, ns={self.num_steps}, "
            f"anneal={'Y' if self.anneal_lr else 'N'}, "
            f"epochs={self.update_epochs}"
        )


def parse_hparams(markdown: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in markdown.splitlines():
        m = _HPARAM_ROW.match(line.strip())
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
                hparams = parse_hparams(v.tensor.string_val[0].decode("utf-8", "ignore"))
            elif v.tag == METRIC:
                steps.append(event.step)
                if v.HasField("tensor") and len(v.tensor.float_val) > 0:
                    returns.append(float(v.tensor.float_val[0]))
                elif v.HasField("simple_value"):
                    returns.append(float(v.simple_value))
                else:
                    returns.append(float("nan"))
    if not hparams or not steps or "seed" not in hparams or "learning_rate" not in hparams:
        return None
    order = np.argsort(steps)
    return RunData(
        event_file=path,
        hparams=hparams,
        steps=np.asarray(steps, dtype=np.int64)[order],
        returns=np.asarray(returns, dtype=np.float64)[order],
    )


def discover_all(runs_dir: Path) -> list[RunData]:
    out: list[RunData] = []
    for ef in sorted(runs_dir.rglob("events.out.tfevents.*")):
        rd = load_event_file(ef)
        if rd is not None:
            out.append(rd)
    return out


def dedupe_by(runs: list[RunData], key) -> list[RunData]:
    """If multiple event files share `key(run)`, keep the longest series."""
    best: dict = {}
    for r in runs:
        k = key(r)
        prev = best.get(k)
        if prev is None or r.returns.size > prev.returns.size:
            best[k] = r
    return list(best.values())


def bin_returns(run: RunData, edges: np.ndarray, fill: bool = True) -> np.ndarray:
    """Mean episodic_return inside each bin defined by `edges` (NaN if empty,
    forward-filled when `fill=True`)."""
    n_bins = edges.size - 1
    out = np.full(n_bins, np.nan)
    if run.steps.size == 0:
        return out
    idx = np.digitize(run.steps, edges) - 1
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = run.returns[mask].mean()
    if fill:
        last = np.nan
        for i in range(n_bins):
            if np.isnan(out[i]):
                out[i] = last
            else:
                last = out[i]
    return out

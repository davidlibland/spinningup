"""Leave-one-out ablation around the successful SB3-replica gSDE config.

Baseline = the exact SB3-zoo Pendulum PPO+gSDE config that replicated SB3
(deterministic eval ~ -114). Each arm flips exactly ONE intervention off,
holding all hyperparameters fixed, and is scored by the DETERMINISTIC
evaluation (action = policy mean, no exploration noise) — the SB3 metric.

Run from this directory: `uv run python ablate_sb3.py`.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PPO = HERE.parent.parent / "algos" / "ppo_gSDE.py"
OUT = HERE / "ablation_sb3"
SEEDS = [101, 102, 103]
N_JOBS = 6

# Exact SB3-replica baseline (the run that hit det-eval ~ -114).
BASE = [
    "--env-id", "Pendulum-v1", "--total-timesteps", "100000",
    "--num-envs", "4", "--num-steps", "1024", "--num-minibatches", "64",
    "--update-epochs", "10", "--learning-rate", "0.001", "--no-anneal-lr",
    "--gamma", "0.9", "--gae-lambda", "0.95", "--clip-coef", "0.2",
    "--no-clip-vloss", "--ent-coef", "0.0", "--vf-coef", "0.5",
    "--max-grad-norm", "0.5", "--use-sde", "--sde-sample-freq", "4",
    "--sde-log-std-init", "0.0", "--no-normalize",
    "--bootstrap-truncation", "--no-sde-learn-features",
    "--eval-episodes", "20",
    "--no-capture-video", "--no-capture-test-video",
]

# Each arm = a list of flag overrides appended after BASE (later flags win
# for tyro), flipping exactly one intervention.
ARMS: dict[str, list[str]] = {
    "A_baseline_SB3replica":      [],
    "B_normalization_ON":         ["--normalize"],
    "C_no_truncation_bootstrap":  ["--no-bootstrap-truncation"],
    "D_value_loss_clipping_ON":   ["--clip-vloss"],
    "E_sde_log_std_init_-2":      ["--sde-log-std-init", "-2.0"],
    "F_sde_learn_features_ON":    ["--sde-learn-features"],
    "G_vanilla_no_gSDE":          ["--no-use-sde"],
}

DET_RE = re.compile(
    r"DET_EVAL episodes=(\d+) mean=(-?[\d.]+) std=(-?[\d.]+) "
    r"min=(-?[\d.]+) max=(-?[\d.]+)"
)


def run_one(arm: str, overrides: list[str], seed: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{arm}_s{seed}.log"
    cmd = (
        ["uv", "run", "python", str(PPO)] + BASE + overrides
        + ["--seed", str(seed), "--exp-name", f"abl-{arm}-s{seed}"]
    )
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    import os
    full_env = {**os.environ, **env}
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                            env=full_env).returncode
    txt = log.read_text()
    m = DET_RE.search(txt)
    det = float(m.group(2)) if m else float("nan")
    return {"arm": arm, "seed": seed, "rc": rc, "det": det,
            "elapsed": time.time() - t0}


def main() -> None:
    if not PPO.exists():
        raise SystemExit(f"missing {PPO}")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, ov, s) for a, ov in ARMS.items() for s in SEEDS]
    print(f"{len(jobs)} runs ({len(ARMS)} arms x {len(SEEDS)} seeds), "
          f"n_jobs={N_JOBS}")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=N_JOBS) as pool:
        futs = [pool.submit(run_one, a, ov, s) for (a, ov, s) in jobs]
        for f in as_completed(futs):
            r = f.result()
            print(f"  done {r['arm']:<26} seed={r['seed']} "
                  f"det_eval={r['det']:8.2f} rc={r['rc']} "
                  f"t={r['elapsed']:.0f}s")
            results.append(r)

    by_arm: dict[str, list[float]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r["det"])

    base = statistics.mean(by_arm["A_baseline_SB3replica"])
    print("\n=== Leave-one-out ablation (DETERMINISTIC eval, mean of 3 seeds) ===")
    print(f"{'arm':<28} {'det_eval_mean':>13} {'per-seed':>26} {'Δ vs base':>10}")
    print("-" * 82)
    for arm in ARMS:
        vals = sorted(by_arm.get(arm, []))
        m = statistics.mean(vals) if vals else float("nan")
        delta = m - base
        ps = " ".join(f"{v:7.1f}" for v in vals)
        tag = "  <-- baseline" if arm == "A_baseline_SB3replica" else ""
        print(f"{arm:<28} {m:>13.2f} {ps:>26} {delta:>+10.1f}{tag}")
    print(f"\nSB3-zoo published reference: ~ -150  (baseline here: {base:.1f})")
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()

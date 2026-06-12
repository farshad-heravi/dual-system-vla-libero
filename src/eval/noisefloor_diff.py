#!/usr/bin/env python
"""Same-condition noise floor: per-episode flip rate between two eval runs.

Both runs MUST use the same --seed (so env object configs are paired and aligned
by episode index). Vary only the realized action noise between them, e.g. with
DUAL_NOISE_SEED=1 vs DUAL_NOISE_SEED=2 (see lerobot_eval_dual.py).

Compares the resulting same-condition flip count against the cross-condition
(e.g. A0-vs-A1) churn reported in the paper: if they are comparable, the churn
is sampling noise; if the floor is ~0, the churn is causal.

Usage:
    python -m src.eval.noisefloor_diff <run1_dir> <run2_dir>
where each dir contains an eval_info.json.
"""
import json
import sys
from math import comb


def succ_by_seed(run_dir: str) -> dict[tuple[int, int], bool]:
    """Map (task_id, episode_index) -> success. Episode index == paired env seed."""
    d = json.load(open(f"{run_dir.rstrip('/')}/eval_info.json"))
    out: dict[tuple[int, int], bool] = {}
    for t in d["per_task"]:
        tid = t["task_id"]
        for i, s in enumerate(t["metrics"]["successes"]):
            out[(tid, i)] = bool(s)
    return out


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    """Exact binomial two-sided McNemar p on discordant pairs (b, c)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def main(run1: str, run2: str) -> None:
    a, b = succ_by_seed(run1), succ_by_seed(run2)
    keys = sorted(set(a) & set(b))
    if not keys:
        sys.exit("No overlapping (task_id, episode) keys — were both runs the same seed/n_episodes?")

    succ1_only = sum(a[k] and not b[k] for k in keys)   # run1 success, run2 fail
    succ2_only = sum(b[k] and not a[k] for k in keys)   # run2 success, run1 fail
    n = len(keys)
    flips = succ1_only + succ2_only
    sr1 = 100 * sum(a[k] for k in keys) / n
    sr2 = 100 * sum(b[k] for k in keys) / n
    p = exact_mcnemar_two_sided(succ1_only, succ2_only)

    print(f"run1={run1}  SR={sr1:.1f}%")
    print(f"run2={run2}  SR={sr2:.1f}%")
    print(f"n={n}  flips={flips} ({100*flips/n:.1f}%)  "
          f"succ1_only={succ1_only}  succ2_only={succ2_only}  McNemar p={p:.3f}")
    print(f"\n=> same-condition noise floor = {flips} discordant pairs "
          f"({100*flips/n:.1f}% of episodes).")
    print("   Compare against the cross-condition (e.g. A0-vs-A1) discordant count.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m src.eval.noisefloor_diff <run1_dir> <run2_dir>")
    main(sys.argv[1], sys.argv[2])

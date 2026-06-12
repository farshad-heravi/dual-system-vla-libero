"""Classify failed A1 episodes into F-Plan / F-Interface / F-Stall / F-Exec."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

CATEGORIES = ("F-Plan", "F-Interface", "F-Stall", "F-Exec")

_EPISODE_GLOB = "episode_*_*.jsonl"
_EPISODE_PREFIX = "episode_"

# Objects that appear in exactly one task — used to generate F-Plan demo traces.
_TASK_EXCLUSIVE_FOREIGN: dict[int, list[str]] = {
    0: ["moka pot", "book", "black bowl"],
    1: ["moka pot", "book", "black bowl"],
    2: ["alphabet soup", "book", "black bowl"],
    3: ["alphabet soup", "moka pot", "butter"],
    4: ["alphabet soup", "moka pot", "book"],
    5: ["alphabet soup", "moka pot", "chocolate pudding"],
    6: ["alphabet soup", "moka pot", "book"],
    7: ["moka pot", "book", "black bowl"],
    8: ["alphabet soup", "book", "black bowl"],
    9: ["alphabet soup", "moka pot", "book"],
}


@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: int
    seed: int
    category: str
    flagged_subtask: str


def load_task_meta(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["tasks"]


def _parse_episode_filename(path: Path) -> Optional[tuple[int, int]]:
    stem = path.stem  # episode_{task_id}_{seed}
    parts = stem.split("_")
    if len(parts) == 3 and parts[0] == "episode":
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return None


def _load_trace(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_foreign_objects(task_id: int, task_meta: dict) -> frozenset[str]:
    """Object names that appear in other tasks but NOT in this task.

    Excludes foreign objects that are substrings of an own object (e.g. "plate"
    is excluded for task_4 because "left plate" is an own object, preventing
    spurious matches via substring containment).
    """
    task_key = f"task_{task_id}"
    own = {o.lower() for o in task_meta[task_key]["objects"]}
    foreign: set[str] = set()
    for key, info in task_meta.items():
        if key == task_key:
            continue
        for obj in info["objects"]:
            lo = obj.lower()
            if lo not in own and not any(lo in own_obj for own_obj in own):
                foreign.add(lo)
    return frozenset(foreign)


def _find_fplan_subtask(
    trace: list[dict], task_id: int, task_meta: dict
) -> str:
    import re

    foreign = _build_foreign_objects(task_id, task_meta)
    # Sort longest first so multi-word phrases match before their substrings.
    foreign_sorted = sorted(foreign, key=len, reverse=True)
    patterns = [re.compile(r"\b" + re.escape(obj) + r"\b") for obj in foreign_sorted]

    for rec in trace:
        sg = (rec.get("current_subtask") or "").lower()
        if any(pat.search(sg) for pat in patterns):
            return rec.get("current_subtask", "")
    return ""


def classify_episode(
    trace: list[dict],
    task_id: int,
    seed: int,
    a4_failed: set[tuple[int, int]],
    task_meta: dict,
) -> tuple[str, str]:
    """Return (category, flagged_subtask). Priority: F-Plan > F-Interface > F-Stall > F-Exec."""
    flagged = _find_fplan_subtask(trace, task_id, task_meta)
    if flagged:
        return "F-Plan", flagged

    if (task_id, seed) in a4_failed:
        return "F-Interface", ""

    max_progress = max((r.get("progress_pct") or 0 for r in trace), default=0)
    if max_progress <= 20:
        return "F-Stall", ""

    return "F-Exec", ""


def _load_a4_failures(a4_dir: Path) -> set[tuple[int, int]]:
    """Return (task_id, seed) pairs where A4 episode ended in failure."""
    failed: set[tuple[int, int]] = set()
    logs = sorted(a4_dir.glob(_EPISODE_GLOB))
    if not logs:
        raise FileNotFoundError(
            f"No episode logs found in {a4_dir}.\n"
            "Expected files named episode_{{task_id}}_{{seed}}.jsonl "
            "(e.g. episode_0_3.jsonl)."
        )
    for log in logs:
        ids = _parse_episode_filename(log)
        if ids is None:
            continue
        task_id, seed = ids
        trace = _load_trace(log)
        if trace and not trace[-1].get("success", True):
            failed.add((task_id, seed))
    return failed


def _validate_log_dir(log_dir: Path, label: str) -> None:
    if not log_dir.exists():
        raise FileNotFoundError(
            f"{label} log directory not found: {log_dir}\n"
            "Expected a directory containing per-episode JSONL files named\n"
            "  episode_{{task_id}}_{{seed}}.jsonl  (e.g. episode_5_2.jsonl)\n"
            "where each line is a step record:\n"
            '  {"step": 25, "current_subtask": "...", "progress_pct": 15, '
            '"should_replan": false, "success": false}\n'
            "and the last line carries the final success value."
        )
    if not any(log_dir.glob(_EPISODE_GLOB)):
        raise FileNotFoundError(
            f"No episode logs matching '{_EPISODE_GLOB}' found in {log_dir}.\n"
            "Generate them by running lerobot_eval_dual.py with per-episode "
            "JSONL logging enabled."
        )


def load_and_classify_a1(
    a1_dir: Path,
    a4_failed: set[tuple[int, int]],
    task_meta: dict,
) -> list[EpisodeRecord]:
    _validate_log_dir(a1_dir, "A1")
    records: list[EpisodeRecord] = []
    for log in sorted(a1_dir.glob(_EPISODE_GLOB)):
        ids = _parse_episode_filename(log)
        if ids is None:
            continue
        task_id, seed = ids
        trace = _load_trace(log)
        if not trace:
            continue
        # Only classify failed episodes.
        if trace[-1].get("success", False):
            continue
        category, flagged = classify_episode(trace, task_id, seed, a4_failed, task_meta)
        records.append(
            EpisodeRecord(
                episode_id=log.stem,
                task_id=task_id,
                seed=seed,
                category=category,
                flagged_subtask=flagged,
            )
        )
    return records


def write_csv(records: list[EpisodeRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "episode_id": r.episode_id,
            "task_id": r.task_id,
            "seed": r.seed,
            "category": r.category,
            "flagged_subtask": r.flagged_subtask,
        }
        for r in records
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Demo mode — generates synthetic records without touching the filesystem
# ---------------------------------------------------------------------------

def _make_demo_trace(
    task_id: int,
    seed: int,
    category: str,
    task_meta: dict,
) -> list[dict]:
    """Build a synthetic step-record trace that will classify as `category`."""
    rng = random.Random(task_id * 1000 + seed)
    task_key = f"task_{task_id}"
    own_objects = task_meta[task_key]["objects"]
    n_steps = 520

    # Subtask sequence drawn from the oracle for this task
    from src.system2.oracle import _decompose  # local import to avoid circular
    subtasks = _decompose(task_meta[task_key]["description"])

    trace: list[dict] = []

    if category == "F-Stall":
        # progress never exceeds 20; sub-goal stays at first subtask
        sg = subtasks[0] if subtasks else "pick up object"
        for step in range(0, n_steps, 45):
            trace.append(
                {
                    "step": step,
                    "current_subtask": sg,
                    "progress_pct": rng.randint(0, 15),
                    "should_replan": False,
                    "success": False,
                }
            )

    elif category == "F-Plan":
        # Inject a foreign object name at step ~225 (mid-episode)
        foreign_objects = _TASK_EXCLUSIVE_FOREIGN.get(task_id, ["moka pot"])
        bad_obj = rng.choice(foreign_objects)
        for i, step in enumerate(range(0, n_steps, 45)):
            sg = subtasks[min(i, len(subtasks) - 1)] if subtasks else "pick up object"
            progress = min(10 + i * 8, 90)
            is_anomaly = step >= 225 and i >= 5
            if is_anomaly:
                sg = f"pick up the {bad_obj}"
            trace.append(
                {
                    "step": step,
                    "current_subtask": sg,
                    "progress_pct": progress,
                    "should_replan": is_anomaly,
                    "success": False,
                }
            )

    elif category in ("F-Interface", "F-Exec"):
        # Reasonable progress (>20), correct sub-goals
        for i, step in enumerate(range(0, n_steps, 45)):
            sg = subtasks[min(i, len(subtasks) - 1)] if subtasks else "pick up object"
            progress = min(10 + i * 7 + rng.randint(0, 5), 95)
            trace.append(
                {
                    "step": step,
                    "current_subtask": sg,
                    "progress_pct": progress,
                    "should_replan": False,
                    "success": False,
                }
            )

    # Final record marks episode outcome
    if trace:
        trace[-1] = {**trace[-1], "success": False}
    return trace


def generate_demo_records(task_meta: dict) -> tuple[list[EpisodeRecord], set[tuple[int, int]]]:
    """Return (classified_records, a4_failed_set) for 300 synthetic episodes."""
    # 30 failed episodes per task; category counts per task (sum=30):
    # F-Exec:11, F-Interface:9, F-Plan:7, F-Stall:3
    per_task_plan = [
        ("F-Exec", 11),
        ("F-Interface", 9),
        ("F-Plan", 7),
        ("F-Stall", 3),
    ]

    a4_failed: set[tuple[int, int]] = set()
    seed_counter: dict[int, int] = {t: 0 for t in range(10)}
    records: list[EpisodeRecord] = []

    for task_id in range(10):
        for category, count in per_task_plan:
            for _ in range(count):
                seed = seed_counter[task_id]
                seed_counter[task_id] += 1
                trace = _make_demo_trace(task_id, seed, category, task_meta)
                # F-Interface episodes need to appear in A4 failures too
                if category == "F-Interface":
                    a4_failed.add((task_id, seed))
                cat_out, flagged = classify_episode(
                    trace, task_id, seed, a4_failed, task_meta
                )
                records.append(
                    EpisodeRecord(
                        episode_id=f"episode_{task_id}_{seed}",
                        task_id=task_id,
                        seed=seed,
                        category=cat_out,
                        flagged_subtask=flagged,
                    )
                )

    return records, a4_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify A1 failed episodes into bottleneck categories."
    )
    p.add_argument("--a1-logs", type=Path, metavar="DIR")
    p.add_argument("--a4-logs", type=Path, metavar="DIR")
    p.add_argument(
        "--task-meta",
        type=Path,
        default=Path("configs/libero10_tasks.yaml"),
        metavar="YAML",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/failure_analysis.csv"),
        metavar="CSV",
    )
    p.add_argument("--out-plots", type=Path, metavar="DIR")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Generate 300 synthetic episodes and run the full pipeline.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        _main(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _main(args: argparse.Namespace) -> None:
    task_meta = load_task_meta(args.task_meta)

    if args.demo:
        records, _ = generate_demo_records(task_meta)
    else:
        if args.a1_logs is None or args.a4_logs is None:
            print(
                "error: --a1-logs and --a4-logs are required unless --demo is set.",
                file=sys.stderr,
            )
            sys.exit(1)
        a4_failed = _load_a4_failures(args.a4_logs)
        records = load_and_classify_a1(args.a1_logs, a4_failed, task_meta)

    write_csv(records, args.out_csv)
    print(f"Wrote {len(records)} records → {args.out_csv}")

    counts = pd.Series([r.category for r in records]).value_counts()
    for cat in CATEGORIES:
        print(f"  {cat}: {counts.get(cat, 0)}")

    if args.out_plots:
        from src.eval.plot_bottleneck import generate_plots
        generate_plots(args.out_csv, args.out_plots)
        print(f"Plots written → {args.out_plots}")


if __name__ == "__main__":
    main()


"""Relabels lerobot/libero_10 with oracle sub-goals for LoRA training.

Each step's task instruction becomes:
    "{original_task}. Currently: {oracle_subgoal_for_step}"

This trains π0's language adapter to follow sub-goal-conditioned instructions,
fixing the OOD suffix degradation observed in A1/A2 ablations.

Usage:
    python scripts/generate_subgoal_dataset.py [--push_to_hub YOUR_HF_USERNAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from src.system2.oracle import ORACLE_SUBGOALS


# ------------------------------------------------------------------
# Sub-goal labeling
# ------------------------------------------------------------------

def oracle_subgoal_for_step(task: str, frame_index: int, ep_length: int) -> str:
    subtasks = ORACLE_SUBGOALS.get(task)
    if not subtasks:
        return task  # unknown task: fall back to original
    n = len(subtasks)
    idx = min(int(frame_index / max(ep_length, 1) * n), n - 1)
    return subtasks[idx]


def relabel_example(example: dict, ep_lengths: dict[int, int]) -> dict:
    task       = example["task"]
    ep_idx     = example["episode_index"]
    frame_idx  = example["frame_index"]
    ep_len     = ep_lengths.get(ep_idx, 520)

    subgoal = oracle_subgoal_for_step(task, frame_idx, ep_len)
    example["task"] = f"{task}. Currently: {subgoal}"
    return example


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", default=None,
                        help="HuggingFace username; if set, pushes relabeled "
                             "dataset as <username>/libero10_subgoal_labeled")
    parser.add_argument("--out_dir", default="./data/libero10_subgoal_labeled")
    args = parser.parse_args()

    print("Loading lerobot/libero_10 ...")
    ds = load_dataset("lerobot/libero_10", split="train")
    print(f"  {len(ds)} frames, columns: {ds.column_names}")

    # Verify expected columns exist
    for col in ("task", "episode_index", "frame_index"):
        if col not in ds.column_names:
            raise RuntimeError(
                f"Column '{col}' not found. Print ds.features and adjust "
                f"relabel_example() to match the actual schema."
            )

    # Build episode-length lookup (max frame_index + 1 per episode)
    print("Computing episode lengths ...")
    ep_lengths: dict[int, int] = {}
    for row in ds:
        ep  = row["episode_index"]
        fi  = row["frame_index"]
        ep_lengths[ep] = max(ep_lengths.get(ep, 0), fi + 1)
    print(f"  {len(ep_lengths)} episodes, "
          f"mean length {sum(ep_lengths.values())/len(ep_lengths):.0f} frames")

    # Relabel
    print("Relabeling ...")
    ds_relabeled = ds.map(
        lambda ex: relabel_example(ex, ep_lengths),
        desc="Oracle sub-goal relabeling",
        num_proc=4,
    )

    # Spot-check: print 5 rows from different episodes
    print("\nSpot-check (5 rows across different episodes):")
    seen = set()
    for row in ds_relabeled:
        ep = row["episode_index"]
        if ep not in seen:
            seen.add(ep)
            print(f"  ep={ep:03d} frame={row['frame_index']:03d}  "
                  f"task={row['task']}")
        if len(seen) == 5:
            break

    # Save
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds_relabeled.save_to_disk(str(out))
    print(f"\nSaved to {out}")

    if args.push_to_hub:
        repo_id = f"{args.push_to_hub}/libero10_subgoal_labeled"
        print(f"Pushing to {repo_id} ...")
        ds_relabeled.push_to_hub(repo_id)
        print("Done.")


if __name__ == "__main__":
    main()

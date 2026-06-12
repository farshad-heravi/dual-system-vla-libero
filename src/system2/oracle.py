"""Oracle S2 runner for A4 ablation.

Returns ground-truth sub-goals derived from the task instruction using
rule-based decomposition of LIBERO pick-and-place tasks. Sub-goals cycle
through the decomposed stages based on elapsed trigger count.
No VLM inference required.
"""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image

from .planner import SubGoal


class OracleS2Runner:
    """A4 ablation: rule-based oracle sub-goal runner.

    Decomposes the task instruction into an ordered list of sub-goals using
    heuristics for LIBERO pick-and-place tasks, then advances through them
    proportionally over the episode using update_obs() trigger counts.

    Interface matches AsyncS2Runner and RandomS2Runner; no background thread.
    """

    def __init__(self, max_steps: int = 520, n_trigger: int = 45):
        # Expected number of update_obs() calls per episode: one at step 0
        # plus periodic triggers.  +2 gives headroom so frac never exceeds 1.
        self._max_triggers = max(1, max_steps // n_trigger) + 2
        self._instruction: str = ""
        self._subtasks: list[str] = []
        self._trigger_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle (no background thread needed)
    # ------------------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset(self, instruction: Optional[str] = None) -> None:
        if instruction is not None:
            self._instruction = instruction
            self._subtasks = _decompose(instruction)
        self._trigger_count = 0

    # ------------------------------------------------------------------
    # S1-facing interface
    # ------------------------------------------------------------------

    def update_obs(self, image: Image.Image) -> None:
        self._trigger_count += 1

    def get_subgoal(self) -> str:
        sg = self.get_subgoal_object()
        if sg is None:
            return self._instruction
        return sg.to_s1_instruction(self._instruction, mode="append")

    def get_subgoal_object(self) -> Optional[SubGoal]:
        if not self._subtasks:
            return None
        idx = self._current_idx()
        subtask = self._subtasks[idx]
        progress_pct = int(100 * idx / len(self._subtasks))
        return SubGoal(
            current_subtask=subtask,
            progress_pct=progress_pct,
            should_replan=False,
            raw_output="[oracle]",
            parse_failed=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_idx(self) -> int:
        n = len(self._subtasks)
        frac = min(self._trigger_count / self._max_triggers, 0.9999)
        return min(int(frac * n), n - 1)


def _decompose(instruction: str) -> list[str]:
    """Split a LIBERO task instruction into ordered sub-goals.

    Patterns are tried most-specific first:
      A. "put both X and Y <prep> Z"       → 4 subtasks  (Tasks 0, 1, 7)
      B. "put both Xs <prep> Z"            → 4 subtasks, noun singularised (Task 8)
      C. "turn on X and put Y on it"       → 3 subtasks  (Task 2)
      D. "put X <prep> Z and close it"     → 3 subtasks  (Tasks 3, 9)
      E. "put X <prep> A and put Y <prep> B" → 4 subtasks (Tasks 4, 6)
      F. "pick up X and place/put it <prep> Y" → 2 subtasks (Task 5)
      G. "put/place X <prep> Y"            → 2 subtasks  (simple)
    Falls back to [instruction] for unrecognised patterns.
    """
    lower = instruction.strip().lower()

    # Preposition alternatives with word boundaries on single-word forms so
    # "in" doesn't match inside "pudding", "on" doesn't match inside "stove".
    # Longer multi-word alternatives come first so "on top of" wins over "on".
    PREP = r"(?:\binto\b|on top of|to the right of|to the left of|\bin\b|\bon\b)"

    # A: "put both X and Y <prep> Z" — two distinct objects, same destination
    m = re.match(rf"(?:put|place) both (.+?) and (.+?) ({PREP}) (.+)", lower)
    if m:
        obj1, obj2, prep, loc = (
            m.group(1).strip(), m.group(2).strip(),
            m.group(3).strip(), m.group(4).strip(),
        )
        return [
            f"pick up {obj1}",
            f"put {obj1} {prep} {loc}",
            f"pick up {obj2}",
            f"put {obj2} {prep} {loc}",
        ]

    # B: "put both Xs <prep> Z" — plural same-type objects (no "and" separator)
    m = re.match(rf"(?:put|place) both (.+?) ({PREP}) (.+)", lower)
    if m:
        noun_pl, prep, loc = (
            m.group(1).strip(), m.group(2).strip(), m.group(3).strip(),
        )
        # Naive singularisation: strip trailing "s" ("moka pots" → "moka pot").
        noun_sg = noun_pl[:-1] if (noun_pl.endswith("s") and not noun_pl.endswith("ss")) else noun_pl
        return [
            f"pick up a {noun_sg}",
            f"put the {noun_sg} {prep} {loc}",
            f"pick up the other {noun_sg}",
            f"put the other {noun_sg} {prep} {loc}",
        ]

    # C: "turn on X and put Y on it"
    m = re.match(r"turn on (.+?) and (?:put|place) (.+?) on it$", lower)
    if m:
        appliance, obj = m.group(1).strip(), m.group(2).strip()
        return [
            f"turn on {appliance}",
            f"pick up {obj}",
            f"put {obj} on {appliance}",
        ]

    # D: "put X <prep> Z and close it"
    m = re.match(rf"(?:put|place) (.+?) ({PREP}) (.+?) and close it$", lower)
    if m:
        obj, prep, loc = (
            m.group(1).strip(), m.group(2).strip(), m.group(3).strip(),
        )
        return [
            f"pick up {obj}",
            f"put {obj} {prep} {loc}",
            f"close {loc}",
        ]

    # E: "put X <prep> A and put Y <prep> B" — two separate placements
    m = re.match(rf"put (.+?) ({PREP}) (.+?) and put (.+?) ({PREP}) (.+)", lower)
    if m:
        obj1, prep1, loc1, obj2, prep2, loc2 = (
            m.group(1).strip(), m.group(2).strip(), m.group(3).strip(),
            m.group(4).strip(), m.group(5).strip(), m.group(6).strip(),
        )
        return [
            f"pick up {obj1}",
            f"put {obj1} {prep1} {loc1}",
            f"pick up {obj2}",
            f"put {obj2} {prep2} {loc2}",
        ]

    # F: "pick up X and place/put it <prep> Y" (original pattern 1, fixed)
    m = re.match(rf"pick up (.+?) and (?:place|put) it ({PREP}) (.+)", lower)
    if m:
        obj, prep, loc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        return [f"pick up {obj}", f"place {obj} {prep} {loc}"]

    # G: "put/place X <prep> Y" (original pattern 2)
    m = re.match(rf"(?:put|place) (.+?) ({PREP}) (.+)", lower)
    if m:
        obj, prep, loc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        return [f"pick up {obj}", f"put {obj} {prep} {loc}"]

    return [instruction.strip()]


class _OracleSubgoalsProxy:
    """Dict-like proxy that decomposes task instructions on demand via _decompose().

    Avoids maintaining a static lookup table while satisfying the ORACLE_SUBGOALS
    interface expected by planner_v2.py.
    """

    def get(self, instruction: str, default=None) -> list[str]:
        return _decompose(instruction)

    def __getitem__(self, instruction: str) -> list[str]:
        return _decompose(instruction)


ORACLE_SUBGOALS = _OracleSubgoalsProxy()

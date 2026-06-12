"""Async System 2 runner.

Wraps Qwen25VLPlanner in a background thread so S1 is never blocked
waiting for S2. Supports all ablation modes:

  A1 (full)   : AsyncS2Runner(planner, frozen=False)
  A2 (frozen) : AsyncS2Runner(planner, frozen=True)  -- fires once at t=0
  A3 (random) : RandomS2Runner(pool)                 -- no planner needed

Goes in src/system2/async_runner.py.
"""

from __future__ import annotations

import queue
import random
import threading
from typing import Optional

from PIL import Image

from .planner import Qwen25VLPlanner, SubGoal


class AsyncS2Runner:
    """Thread-safe System 2 wrapper.

    Lifecycle per episode:
        runner.reset(instruction)   # sync instruction to current task
        runner.start()              # (call once globally, not per episode)
        ...eval loop...
        runner.update_obs(img)      # every n_trigger steps
        subgoal = runner.get_subgoal()  # every step
        runner.stop()               # at end of full eval run
    """

    def __init__(
        self,
        planner: Qwen25VLPlanner,
        instruction: str = "",
        mode: str = "append",
        frozen: bool = False,
    ):
        """
        Args:
            planner: loaded Qwen25VLPlanner (shared across episodes).
            instruction: task instruction, updated each episode via reset().
            mode: 'append' keeps original task + sub-goal suffix (safer for π0).
                  'replace' passes sub-goal only (use for A1-replace ablation).
            frozen: A2 ablation. S2 fires on the first frame then ignores all
                    subsequent update_obs() calls for the rest of the episode.
        """
        self.planner = planner
        self.instruction = instruction
        self.mode = mode
        self.frozen = frozen

        self._lock = threading.Lock()
        self._latest: Optional[SubGoal] = None
        self._prev_subtask: str = "none"
        self._has_fired: bool = False

        # maxsize=1: always process the newest frame; drop stale ones.
        self._obs_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch background inference thread. Call once before the eval loop."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal thread to exit and wait for it. Call after eval loop ends."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def reset(self, instruction: Optional[str] = None) -> None:
        """Reset state between episodes. Thread-safe; does not stop the thread."""
        if instruction is not None:
            self.instruction = instruction
        with self._lock:
            self._latest = None
            self._has_fired = False
        self._prev_subtask = "none"
        # Drain stale frames from the previous episode.
        while not self._obs_queue.empty():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # S1-facing interface (called from the eval loop)
    # ------------------------------------------------------------------

    def update_obs(self, image: Image.Image) -> None:
        """Push a new observation frame. Non-blocking; always keeps the newest."""
        if self.frozen:
            with self._lock:
                if self._has_fired:
                    return  # A2: ignore frames after the first sub-goal
        if self._obs_queue.full():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._obs_queue.put_nowait(image)
        except queue.Full:
            pass

    def get_subgoal(self) -> str:
        """Non-blocking read. Returns original instruction until S2 first fires."""
        with self._lock:
            if self._latest is None:
                return self.instruction
            return self._latest.to_s1_instruction(self.instruction, mode=self.mode)

    def get_subgoal_object(self) -> Optional[SubGoal]:
        """Full SubGoal for logging. Returns None before first inference."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                image = self._obs_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            subgoal = self.planner(image, self.instruction, self._prev_subtask)

            if not subgoal.parse_failed:
                self._prev_subtask = subgoal.current_subtask

            with self._lock:
                self._latest = subgoal
                self._has_fired = True


class RandomS2Runner:
    """A3 ablation: returns randomly shuffled sub-goals from other tasks.

    No background thread needed — no inference is performed.
    Implements the same interface as AsyncS2Runner so it drops in directly.
    """

    def __init__(self, pool: list[str]):
        """
        Args:
            pool: sub-goal strings drawn from OTHER libero_10 task instructions.
                  Build with: [t for t in all_tasks if t != current_task_instruction]
        """
        if not pool:
            raise ValueError("RandomS2Runner pool must not be empty.")
        self._pool = pool

    # Lifecycle stubs (no-ops)
    def start(self) -> None: pass
    def stop(self) -> None: pass

    def reset(self, instruction: Optional[str] = None) -> None: pass

    def update_obs(self, image: Image.Image) -> None: pass

    def get_subgoal(self) -> str:
        return random.choice(self._pool)

    def get_subgoal_object(self) -> None:
        return None

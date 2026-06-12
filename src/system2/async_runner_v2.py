"""Async System 2 runner for A5 VLM-gated sub-goal advancement.

Wraps Qwen25VLPlannerV2 in the same background-thread interface as
AsyncS2Runner so it drops into lerobot_eval_dual.py without changes.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Optional

from PIL import Image

from .planner_v2 import Qwen25VLPlannerV2, SubGoalV2, _PRESEED_MARKER


class AsyncS2RunnerV2:
    """Thread-safe wrapper for Qwen25VLPlannerV2 (A5 ablation).

    Lifecycle per episode:
        runner.reset(instruction)       # sync instruction, reset planner state
        runner.start()                  # call once globally, not per episode
        ...eval loop...
        runner.update_obs(img)          # every n_trigger steps
        subgoal = runner.get_subgoal()  # every step
        runner.stop()                   # at end of full eval run
    """

    def __init__(
        self,
        planner: Qwen25VLPlannerV2,
        instruction: str = "",
        mode: str = "append",
        n_frames: int = 1,
    ):
        self._planner = planner
        self.instruction = instruction
        self.mode = mode

        self._lock = threading.Lock()
        self._latest: Optional[SubGoalV2] = None
        self._prev_subtask: str = "none"

        # Rolling buffer of the last n_frames observations.
        self._frame_buffer: deque = deque(maxlen=n_frames)
        # Signal-only queue: payload is always None; data lives in _frame_buffer.
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
        self._planner.reset(self.instruction)
        # Pre-seed with the first oracle subtask so π0 gets conditioning from
        # step 0 rather than waiting for the first slow inference call to land.
        with self._lock:
            self._latest = SubGoalV2(
                current_subtask=self._planner.first_subtask(),
                progress_pct=0,
                should_replan=False,
                raw_output=_PRESEED_MARKER,
                parse_failed=False,
            )
        self._prev_subtask = "none"
        self._frame_buffer.clear()
        while not self._obs_queue.empty():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # S1-facing interface (called from the eval loop)
    # ------------------------------------------------------------------

    def update_obs(self, image: Image.Image) -> None:
        """Append frame to the rolling buffer and signal the background thread."""
        self._frame_buffer.append(image)
        # Drop any stale unprocessed signal before sending the new one.
        if self._obs_queue.full():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._obs_queue.put_nowait(None)
        except queue.Full:
            pass

    def get_subgoal(self) -> str:
        """Non-blocking read. Returns original instruction until S2 first fires."""
        with self._lock:
            if self._latest is None:
                return self.instruction
            return self._latest.to_s1_instruction(self.instruction, mode=self.mode)

    def get_subgoal_object(self) -> Optional[SubGoalV2]:
        """Full SubGoalV2 for logging. Returns None before first inference."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._obs_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frames = list(self._frame_buffer)
            if not frames:
                continue

            try:
                subgoal = self._planner(frames, self.instruction, self._prev_subtask)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[AsyncS2RunnerV2] inference error — will retry on next obs: {type(exc).__name__}: {exc}", flush=True)
                continue

            if not subgoal.parse_failed:
                self._prev_subtask = subgoal.current_subtask

            with self._lock:
                self._latest = subgoal

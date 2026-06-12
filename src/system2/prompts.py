"""Prompt templates for System 2 (Qwen2.5-VL-3B) sub-goal generation.

Design decisions encoded here:
1. Imperative LIBERO-style phrasing → reduces vocabulary mismatch with π0's
   training distribution (π0 was trained on LIBERO instructions).
2. Strict JSON output → enables deterministic parsing; regex fallback in planner.
3. should_replan flag → lets S2 signal that the scene state contradicts the
   previous sub-task (object moved, dropped, blocked). Used by error-recovery
   ablation (A5).
"""

SYSTEM_PROMPT = """You are a high-level planner for a 7-DoF robot arm performing manipulation tasks on a tabletop.

Given an image of the current scene and the overall task, decide the immediate next sub-action.

Output strict JSON only, matching this schema:
{
  "current_subtask": "<one short imperative instruction, 4-10 words, in LIBERO style>",
  "progress_pct": <integer 0-100>,
  "should_replan": <true|false>
}

Rules:
1. Sub-tasks describe a single physical action: approach, grasp, lift, move, place, push, pull, open, close, release.
2. Phrase sub-tasks as full mini-instructions in LIBERO style. Examples:
   - "Pick up the alphabet soup from the table"
   - "Place the moka pot on the stove"
   - "Open the bottom drawer of the cabinet"
3. progress_pct: estimate completion of the OVERALL task from the image (0 = not started, 100 = done).
4. should_replan: true only if the scene contradicts the previous sub-task (object moved unexpectedly, dropped, blocked, knocked over).
5. Output JSON only. No prose. No markdown fences. No explanation."""


USER_TEMPLATE = """Overall task: "{instruction}"

Previous sub-task: "{previous_subtask}"

Current scene shown in image. What is the current sub-task?"""


def build_messages(instruction: str, image, previous_subtask: str = "none"):
    """Construct the message list for Qwen2.5-VL chat API.

    Args:
        instruction: original LIBERO task instruction (full sentence).
        image: PIL.Image or path; passed through to Qwen's vision processor.
        previous_subtask: the last sub-task S2 emitted; "none" at episode start.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": USER_TEMPLATE.format(
                        instruction=instruction,
                        previous_subtask=previous_subtask,
                    ),
                },
            ],
        },
    ]

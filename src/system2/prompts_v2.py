"""Prompt templates for A5 VLM-gated completion detection.

Unlike prompts.py (sub-goal generation), these templates ask Qwen to act as
a binary completion detector: given the current scene image and the active
sub-task, decide whether that sub-task is visually finished.
"""

SYSTEM_PROMPT_V2 = """You verify sub-task completion for a 7-DoF robot arm on a tabletop (LIBERO simulation).
You receive the sub-task text and one or more frames. If multiple frames: Image 1 = reference (state when sub-task began), later images = chronological, last image = current state. Judge ONLY the last image; earlier frames give context for what changed.

Procedure (fill JSON fields in this exact order, they force correct reasoning):
1. "criterion": one sentence stating what visible end-state defines success for THIS sub-task. Pick-place: object resting at target, gripper open and clear. Open/close: drawer or door at end position. Turn-on: knob/switch rotated, indicator state changed. Push: object at named region.
2. "evidence": one short phrase describing what the last frame actually shows about the target object and gripper.
3. "object_at_goal": true/false, target object visibly in its goal state in the last frame.
4. "gripper_disengaged": true/false, gripper released and clear (set true automatically for sub-tasks with no grasp, e.g. push, turn knob).
5. "subtask_complete": true ONLY if both object_at_goal and gripper_disengaged are true. Otherwise false. When evidence is ambiguous or occluded, output false.
6. "should_replan": true ONLY if the target object is absent, toppled, or fell out of reach. Robot still mid-motion is NOT a replan condition.
7. "confidence": integer 0-100. Your certainty that subtask_complete is correct. Use 90+ only when visual evidence is unambiguous (object clearly at goal, gripper clearly open and clear). Use 50-89 when likely but slightly occluded. Use 0-49 when uncertain.

Output strict JSON only, no markdown, no extra text:
{"criterion": "...", "visual_evidence": "...", "object_at_goal": true|false, "gripper_disengaged": true|false, "subtask_complete": true|false, "should_replan": true|false, "confidence": 0}"""

USER_TEMPLATE_V3 = """Overall task: "{instruction}"
Current sub-task: "{current_subtask}"
Image 1 is the reference frame (sub-task start). Image {n} is the current frame. Apply the procedure and output JSON."""

USER_TEMPLATE_V2 = """Overall task: "{instruction}"

Current sub-task: "{current_subtask}"

Does the image show that the current sub-task is complete?"""

USER_TEMPLATE_V2_WITH_REF = """Overall task: "{instruction}"

Current sub-task: "{current_subtask}"

The first image is the reference frame (scene state when this sub-task began). The second image is the current frame.
Compare them and decide: does the current frame show that the sub-task is complete?"""

USER_TEMPLATE_V2_MULTIFRAME = """Overall task: "{instruction}"

Current sub-task: "{current_subtask}"

You are given {n_frames} consecutive frames in chronological order (earliest to most recent).
Examine the motion and state changes across the sequence. Does the final frame show that the current sub-task is complete?"""

USER_TEMPLATE_V2_MULTIFRAME_WITH_REF = """Overall task: "{instruction}"

Current sub-task: "{current_subtask}"

The first image is the reference frame (scene state when this sub-task began). The following {n_frames} images are consecutive frames in chronological order, with the last being the current frame.
Use the transition from the reference through the sequence to assess completion — look for evidence of motion (object lifted off surface, gripper closed around object, object arriving at target)."""


def build_messages_v2(instruction: str, images, current_subtask: str, ref_image=None) -> list:
    """Construct the message list for Qwen2.5-VL completion-detection call.

    Args:
        instruction: original LIBERO task instruction.
        images: current frame(s) — PIL.Image, path, or list thereof.
        current_subtask: the sub-task being checked for completion.
        ref_image: optional reference frame (scene at subtask start) — PIL.Image or path.
    """
    if not isinstance(images, list):
        images = [images]
    n = len(images)

    content = []
    if ref_image is not None:
        content.append({"type": "image", "image": ref_image})
    for img in images:
        content.append({"type": "image", "image": img})

    if n == 1 and ref_image is None:
        text = USER_TEMPLATE_V2.format(instruction=instruction, current_subtask=current_subtask)
    elif n == 1:
        text = USER_TEMPLATE_V2_WITH_REF.format(instruction=instruction, current_subtask=current_subtask)
    elif ref_image is None:
        text = USER_TEMPLATE_V2_MULTIFRAME.format(instruction=instruction, current_subtask=current_subtask, n_frames=n)
    else:
        text = USER_TEMPLATE_V2_MULTIFRAME_WITH_REF.format(instruction=instruction, current_subtask=current_subtask, n_frames=n)

    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": content},
    ]


def build_prompt_internvl_v2(instruction: str, current_subtask: str, ref_image=None, n_frames: int = 1) -> tuple:
    """Return (system_message, question) for InternVL completion-detection call.

    The question includes <image> tokens required by InternVL's chat API.
    Caller must stack pixel_values as (ref, frame_0, ..., frame_{n-1}) along dim=0.

    Args:
        instruction: original LIBERO task instruction.
        current_subtask: the sub-task being checked for completion.
        ref_image: optional reference frame.
        n_frames: number of consecutive observation frames (not counting ref_image).

    Returns:
        (system_message, question) — both plain strings.
    """
    n_tokens = (1 if ref_image is not None else 0) + n_frames
    image_tokens = "<image>\n" * n_tokens

    if n_frames == 1 and ref_image is None:
        text = USER_TEMPLATE_V2.format(instruction=instruction, current_subtask=current_subtask)
    elif n_frames == 1:
        text = USER_TEMPLATE_V2_WITH_REF.format(instruction=instruction, current_subtask=current_subtask)
    elif ref_image is None:
        text = USER_TEMPLATE_V2_MULTIFRAME.format(instruction=instruction, current_subtask=current_subtask, n_frames=n_frames)
    else:
        text = USER_TEMPLATE_V2_MULTIFRAME_WITH_REF.format(instruction=instruction, current_subtask=current_subtask, n_frames=n_frames)

    return SYSTEM_PROMPT_V2, image_tokens + text

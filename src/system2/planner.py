"""Qwen2.5-VL-3B wrapper for System 2 sub-goal generation.

Goes in src/system2/planner.py per the repo structure in the plan.

Usage:
    >>> planner = Qwen25VLPlanner()
    >>> subgoal = planner(image=pil_image, instruction="Put the bowl on the plate")
    >>> print(subgoal.current_subtask)
    "Pick up the bowl from the table"
    >>> s1_input = subgoal.to_s1_instruction(original_instruction, mode="append")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Union

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from .prompts import build_messages


@dataclass
class SubGoal:
    """Structured output of System 2."""

    current_subtask: str
    progress_pct: int
    should_replan: bool
    raw_output: str
    parse_failed: bool = False

    def to_s1_instruction(
        self, original_instruction: str, mode: str = "append"
    ) -> str:
        """Format sub-goal for injection into System 1's language input.

        mode='replace': use sub-goal only. Risky if π0 wasn't trained on this
            phrasing. Use this mode to ablate the value of preserving the
            original task context.
        mode='append': preserve original task + add current focus. Safer default,
            keeps π0's language embedding close to its training distribution.
        """
        if mode == "replace":
            return self.current_subtask
        return f"{original_instruction}. Currently: {self.current_subtask}"


class Qwen25VLPlanner:
    """Frozen System 2 planner. Single-threaded synchronous interface.

    For async usage in the eval loop, wrap an instance in a background thread
    (see src/system2/async_runner.py).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda:0",
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = True,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        # Low temperature for structured output stability. We are NOT looking
        # for creative outputs; we want consistent JSON-parseable sub-goals.
        self.temperature = temperature

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            if load_in_4bit
            else None
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation="flash_attention_2",
                quantization_config=bnb_config,
            )
            .eval()
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[Image.Image, str],
        instruction: str,
        previous_subtask: str = "none",
    ) -> SubGoal:
        messages = build_messages(instruction, image, previous_subtask)

        # Build chat-template text and extract images for Qwen's processor.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
        )

        # Trim the prompt tokens; decode only the new generation.
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        return _parse(output_text)


def _parse(raw: str) -> SubGoal:
    """Parse Qwen output into a SubGoal. Fall back gracefully on malformed JSON.

    Qwen2.5-VL is usually reliable on JSON but occasionally:
    - Wraps output in ```json fences despite instruction.
    - Adds a leading "Output:" prefix.
    - Trims a trailing comma.
    We strip these defensively before json.loads, then fall back to regex.
    """
    cleaned = raw.strip()
    # Strip markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Strip common prefixes.
    cleaned = re.sub(r"^Output:\s*", "", cleaned, flags=re.IGNORECASE)
    # Extract the first {...} object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
        return SubGoal(
            current_subtask=str(data.get("current_subtask", "")).strip(),
            progress_pct=int(data.get("progress_pct", 0)),
            should_replan=bool(data.get("should_replan", False)),
            raw_output=raw,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        # Regex fallback: pull whatever looks like a subtask from quotes.
        subtask_match = re.search(r'"current_subtask"\s*:\s*"([^"]+)"', cleaned)
        progress_match = re.search(r'"progress_pct"\s*:\s*(\d+)', cleaned)
        replan_match = re.search(r'"should_replan"\s*:\s*(true|false)', cleaned)
        return SubGoal(
            current_subtask=subtask_match.group(1) if subtask_match else "",
            progress_pct=int(progress_match.group(1)) if progress_match else 0,
            should_replan=(replan_match.group(1) == "true") if replan_match else False,
            raw_output=raw,
            parse_failed=True,
        )

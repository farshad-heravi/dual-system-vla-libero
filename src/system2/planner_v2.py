"""VLM wrapper for A5 VLM-gated sub-goal advancement.

Supports two backends selected automatically from the model_id:
  - Qwen2.5-VL   (default)  — "Qwen/Qwen2.5-VL-3B-Instruct"
  - InternVL2.5             — "OpenGVLab/InternVL2_5-4B"

The class name is kept as Qwen25VLPlannerV2 for backward compatibility with
async_runner_v2.py and __init__.py imports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

try:
    from .oracle import ORACLE_SUBGOALS
except ImportError:
    ORACLE_SUBGOALS = {}

from .prompts_v2 import build_messages_v2, build_prompt_internvl_v2


# ---------------------------------------------------------------------------
# InternVL image preprocessing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _make_internvl_transform(image_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _preprocess_internvl(
    image: Union[Image.Image, str], image_size: int = 448
) -> torch.Tensor:
    """Single-tile preprocessing for InternVL (no dynamic tiling).

    Robot-arm images are single-view scenes where dynamic tiling adds
    complexity without benefit.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    return _make_internvl_transform(image_size)(image).unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_backend(model_id: str) -> str:
    if "internvl" in model_id.lower():
        return "internvl"
    return "qwen"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    """Structured output of the VLM completion-detection call."""

    subtask_complete: bool
    confidence: int
    visual_evidence: str
    should_replan: bool
    raw_output: str
    parse_failed: bool = False


@dataclass
class SubGoalV2:
    """Structured sub-goal state returned by Qwen25VLPlannerV2."""

    current_subtask: str
    progress_pct: int
    should_replan: bool
    raw_output: str
    parse_failed: bool = False

    def to_s1_instruction(self, original_instruction: str, mode: str = "append") -> str:
        if mode == "replace":
            return self.current_subtask
        return f"{original_instruction}. Currently: {self.current_subtask}"


_PRESEED_MARKER = "[pre-seeded]"

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Qwen25VLPlannerV2:
    """VLM-gated sub-goal planner for A5.

    Holds an oracle sub-goal list and calls a VLM as a completion detector to
    decide when to advance. Falls back to time-based advancement after
    max_calls_per_subtask consecutive calls on one sub-task.

    Supported models (auto-detected from model_id):
        - Qwen/Qwen2.5-VL-3B-Instruct  (default)
        - OpenGVLab/InternVL2_5-4B
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        dtype: torch.dtype = torch.float16,
        confidence_threshold: int = 70,
        max_calls_per_subtask: int = 60,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.max_calls_per_subtask = max_calls_per_subtask
        self._backend = _detect_backend(model_id)

        if self._backend == "internvl":
            self._load_internvl(model_id, device, dtype)
        else:
            self._load_qwen(model_id, device, dtype)

        # Episode state — initialised properly via reset()
        self._subtasks: list[str] = []
        self._subtask_idx: int = 0
        self._calls_on_idx: int = 0
        self._instruction: str = ""
        self._ref_image: Optional[Union[Image.Image, str]] = None

    def _load_qwen(self, model_id: str, device: str, dtype: torch.dtype) -> None:
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
            .eval()
        )

    def _load_internvl(self, model_id: str, device: str, dtype: torch.dtype) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        ).eval()

    def reset(self, instruction: str) -> None:
        """Initialise episode state. Call at the start of every episode."""
        self._instruction = instruction
        self._subtasks = ORACLE_SUBGOALS.get(instruction, [instruction])
        self._subtask_idx = 0
        self._calls_on_idx = 0
        self._ref_image = None

    def first_subtask(self) -> str:
        """Return the first oracle subtask for this instruction (pre-seed use)."""
        return self._subtasks[0] if self._subtasks else self._instruction

    @torch.inference_mode()
    def __call__(
        self,
        images: Union[Image.Image, str, list],
        instruction: str,
        previous_subtask: str = "none",
    ) -> SubGoalV2:
        # Guard: auto-reset when a new episode starts (instruction changed).
        if instruction != self._instruction:
            self.reset(instruction)

        if not isinstance(images, list):
            images = [images]

        # Capture the reference frame on the first call for this subtask.
        if self._ref_image is None:
            self._ref_image = images[0]

        current_subtask = self._subtasks[self._subtask_idx]

        if self._backend == "internvl":
            raw = self._infer_internvl(images, instruction, current_subtask, self._ref_image)
        else:
            raw = self._infer_qwen(images, instruction, current_subtask, self._ref_image)

        result = _parse_completion(raw)
        self._calls_on_idx += 1

        # When parse failed, confidence field is absent (defaults to 0), so don't
        # gate on threshold — the regex fallback already extracted subtask_complete.
        confidence_ok = result.parse_failed or (result.confidence >= self.confidence_threshold)
        should_advance = (
            (result.subtask_complete and confidence_ok)
            or (self._calls_on_idx >= self.max_calls_per_subtask)
        )
        max_idx = len(self._subtasks) - 1
        if should_advance and self._subtask_idx < max_idx:
            self._subtask_idx += 1
            self._calls_on_idx = 0
            self._ref_image = None  # reset so next call captures the new subtask's reference

        current_subtask = self._subtasks[self._subtask_idx]
        progress_pct = int(self._subtask_idx / len(self._subtasks) * 100)

        return SubGoalV2(
            current_subtask=current_subtask,
            progress_pct=progress_pct,
            should_replan=result.should_replan,
            raw_output=raw,
            parse_failed=result.parse_failed,
        )

    def _infer_qwen(
        self,
        images: list,
        instruction: str,
        current_subtask: str,
        ref_image: Optional[Union[Image.Image, str]] = None,
    ) -> str:
        messages = build_messages_v2(instruction, images, current_subtask, ref_image=ref_image)
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

        do_sample = self.temperature > 0
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else 1.0,
        )
        generated_ids = self.model.generate(**inputs, generation_config=generation_config)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _infer_internvl(
        self,
        images: list,
        instruction: str,
        current_subtask: str,
        ref_image: Optional[Union[Image.Image, str]] = None,
    ) -> str:
        system_message, question = build_prompt_internvl_v2(
            instruction, current_subtask, ref_image=ref_image, n_frames=len(images)
        )
        pixel_tensors = []
        if ref_image is not None:
            pixel_tensors.append(_preprocess_internvl(ref_image).to(device=self.device, dtype=self.model.dtype))
        for img in images:
            pixel_tensors.append(_preprocess_internvl(img).to(device=self.device, dtype=self.model.dtype))
        pixel_values = torch.cat(pixel_tensors, dim=0)
        do_sample = self.temperature > 0
        generation_config: dict = dict(max_new_tokens=self.max_new_tokens, do_sample=do_sample)
        if do_sample:
            generation_config["temperature"] = self.temperature
        if system_message:
            question = f"{system_message}\n\n{question}"
        return self.model.chat(self.tokenizer, pixel_values, question, generation_config)


# ---------------------------------------------------------------------------
# JSON parser (shared by both backends)
# ---------------------------------------------------------------------------

def _parse_completion(raw: str) -> CompletionResult:
    """Parse VLM completion-detection output into a CompletionResult."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^Output:\s*", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start: end + 1]

    try:
        data = json.loads(cleaned)
        return CompletionResult(
            subtask_complete=bool(data.get("subtask_complete", False)),
            confidence=int(data.get("confidence", 0)),
            visual_evidence=str(data.get("visual_evidence") or data.get("evidence", "")).strip(),
            should_replan=bool(data.get("should_replan", False)),
            raw_output=raw,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        complete_match = re.search(r'"subtask_complete"\s*:\s*(true|false)', cleaned)
        confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', cleaned)
        replan_match = re.search(r'"should_replan"\s*:\s*(true|false)', cleaned)
        evidence_match = re.search(r'"visual_evidence"\s*:\s*"([^"]+)"', cleaned)
        return CompletionResult(
            subtask_complete=(complete_match.group(1) == "true") if complete_match else False,
            confidence=int(confidence_match.group(1)) if confidence_match else 0,
            visual_evidence=evidence_match.group(1) if evidence_match else "",
            should_replan=(replan_match.group(1) == "true") if replan_match else False,
            raw_output=raw,
            parse_failed=True,
        )

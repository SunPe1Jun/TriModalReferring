#!/usr/bin/env python3
"""Render controlled Experiment 3 model-input ablations from a frozen manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from point_grounding_common import normalize_text


ABLATION_VARIANTS = (
    "no_visual",
    "no_gaze",
    "no_hand",
    "no_hand_strict",
    "no_gaze_hand",
    "no_instruction",
)


def _json(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(normalize_text(value))
    except json.JSONDecodeError:
        return fallback


def render_prompt(template: str, row: Mapping[str, str], variant: str) -> str:
    """Mask fields in the frozen per-sample prompt without reformatting it.

    ``template`` remains an explicit argument so the run configuration records
    the same template as the baseline. The frozen expanded prompt in the
    manifest is authoritative and prevents unrelated wording/order changes.
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant}")
    del template
    source = normalize_text(row.get("prompt_text"))
    if not source:
        raise ValueError("manifest row has no frozen prompt_text")

    remove_gaze = variant in {"no_gaze", "no_gaze_hand"}
    remove_hand = variant in {"no_hand", "no_hand_strict", "no_gaze_hand"}
    prompt = source
    if variant == "no_instruction":
        prompt = re.sub(r"instruction_text:.*?(?=utterance_text:|Scene bounds:)", "", prompt)
        prompt = re.sub(r"utterance_text:.*?(?=Scene bounds:)", "", prompt)
    if variant == "no_visual":
        prompt = re.sub(
            r"(?<=image: )\S+?(?=\s+relative_sample_time_seconds:)",
            "[visual_evidence_withheld]",
            prompt,
        )
    if remove_gaze or remove_hand:
        # Selection score/reason are derived from both behavioral cues.
        prompt = re.sub(r"\s+selection_score:.*?(?=\s+camera_context_world:)", "", prompt)
    if remove_gaze:
        prompt = re.sub(r"\s+gaze_valid:.*?(?=\s+hand_valid:)", "", prompt)
        prompt = re.sub(
            r"\s+primary_copyable_gaze_point_hypotheses:.*?(?=\s+secondary_copyable_hand_point_hypotheses:)",
            "",
            prompt,
        )
        prompt = re.sub(r"\s+camera_to_gaze_hit_distance:\s+\S+", "", prompt)
        prompt = re.sub(r"\s+gaze_to_hand_hit_distance:\s+\S+", "", prompt)
    if remove_hand:
        prompt = re.sub(
            r"\s+hand_valid:.*?(?=\s+(?:primary_copyable_gaze_point_hypotheses:|secondary_copyable_hand_point_hypotheses:))",
            "",
            prompt,
        )
        prompt = re.sub(
            r"\s+secondary_copyable_hand_point_hypotheses:.*?(?=\s+scale_cues_world_units:)",
            "",
            prompt,
        )
        prompt = re.sub(r"\s+camera_to_hand_hit_distance:\s+\S+", "", prompt)
        prompt = re.sub(r"\s+gaze_to_hand_hit_distance:\s+\S+", "", prompt)
    return prompt


def frame_paths(row: Mapping[str, str], variant: str) -> List[Path]:
    if variant == "no_visual":
        return []
    paths = _json(row.get("frame_paths_json"), [])
    return [Path(str(path)) for path in paths] if isinstance(paths, list) else []


def audit_prompt(prompt: str, variant: str) -> Dict[str, Any]:
    lowered = prompt.lower()
    return {
        "variant": variant,
        "uses_images": variant != "no_visual",
        "has_visual_path": "image: [visual_evidence_withheld]" not in prompt and "image:" in prompt,
        "has_gaze_payload": bool(re.search(r"\bP\d+_GAZE:\s*\[", prompt)),
        "has_hand_payload": bool(re.search(r"\bP\d+_HAND:\s*\[", prompt)),
        "has_instruction_value": bool(re.search(r"instruction_text:\s*\S", prompt)),
        "contains_nan_or_inf": bool(re.search(r"\b(?:nan|[+-]?inf(?:inity)?)\b", lowered)),
    }

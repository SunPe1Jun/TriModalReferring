import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation_inputs import audit_prompt, frame_paths, render_prompt
from point_grounding_common import read_csv_rows


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROW = read_csv_rows(ROOT / "exam3_point_grounding/outputs_full_v9_20260709/manifest.csv")[0]
TEMPLATE = (ROOT / "exam3_point_grounding/prompts/qwen3vl_point_grounding.md").read_text(encoding="utf-8")


def test_variant_prompts_have_no_unfilled_blocks():
    for variant in ("no_visual", "no_gaze", "no_hand", "no_hand_strict", "no_gaze_hand", "no_instruction"):
        prompt = render_prompt(TEMPLATE, MANIFEST_ROW, variant)
        assert "[EVENT_BLOCK]" not in prompt
        assert "[SCENE_BOUNDS_BLOCK]" not in prompt
        assert "[EVIDENCE_BLOCK]" not in prompt
        assert not audit_prompt(prompt, variant)["contains_nan_or_inf"]


def test_no_visual_removes_image_inputs():
    prompt = render_prompt(TEMPLATE, MANIFEST_ROW, "no_visual")
    assert frame_paths(MANIFEST_ROW, "no_visual") == []
    assert "[visual_evidence_withheld]" in prompt
    assert not audit_prompt(prompt, "no_visual")["has_visual_path"]


def test_no_gaze_removes_all_gaze_values():
    prompt = render_prompt(TEMPLATE, MANIFEST_ROW, "no_gaze")
    assert "gaze_hit_world_ray_endpoint:" not in prompt
    assert "gaze_origin_world:" not in prompt
    assert "gaze_direction_world_unit_vector_do_not_copy:" not in prompt
    assert not audit_prompt(prompt, "no_gaze")["has_gaze_payload"]
    assert audit_prompt(prompt, "no_gaze")["has_hand_payload"]


def test_no_hand_removes_all_hand_values():
    for variant in ("no_hand", "no_hand_strict"):
        prompt = render_prompt(TEMPLATE, MANIFEST_ROW, variant)
        assert "hand_hit_world_ray_endpoint:" not in prompt
        assert "hand_origin_world:" not in prompt
        assert "hand_direction_world_unit_vector_do_not_copy:" not in prompt
        assert not audit_prompt(prompt, variant)["has_hand_payload"]
        assert audit_prompt(prompt, variant)["has_gaze_payload"]


def test_no_gaze_hand_removes_both_behavioral_payloads():
    audit = audit_prompt(render_prompt(TEMPLATE, MANIFEST_ROW, "no_gaze_hand"), "no_gaze_hand")
    assert not audit["has_gaze_payload"]
    assert not audit["has_hand_payload"]


def test_no_instruction_removes_language_value():
    prompt = render_prompt(TEMPLATE, MANIFEST_ROW, "no_instruction")
    assert "instruction_text:" not in prompt
    assert "utterance_text:" not in prompt
    assert MANIFEST_ROW["instruction"] not in prompt
    assert not audit_prompt(prompt, "no_instruction")["has_instruction_value"]


def test_renderer_preserves_all_unmasked_lines_verbatim():
    source = MANIFEST_ROW["prompt_text"].strip()
    for variant in ("no_gaze", "no_hand", "no_hand_strict", "no_gaze_hand", "no_instruction"):
        rendered = render_prompt(TEMPLATE, MANIFEST_ROW, variant)
        cursor = 0
        for character in rendered:
            while cursor < len(source) and source[cursor] != character:
                cursor += 1
            assert cursor < len(source), (variant, character)
            cursor += 1
    visual = render_prompt(TEMPLATE, MANIFEST_ROW, "no_visual")
    assert visual.count("image: [visual_evidence_withheld]") == 3
    assert "/evidence_frames/" not in visual


if __name__ == "__main__":
    test_variant_prompts_have_no_unfilled_blocks()
    test_no_visual_removes_image_inputs()
    test_no_gaze_removes_all_gaze_values()
    test_no_hand_removes_all_hand_values()
    test_no_gaze_hand_removes_both_behavioral_payloads()
    test_no_instruction_removes_language_value()
    test_renderer_preserves_all_unmasked_lines_verbatim()

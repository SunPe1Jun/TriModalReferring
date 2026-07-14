import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_point_grounding_manifest import build_prompt


def test_prompt_does_not_use_target_description_or_gt_values():
    template = Path("exam3_point_grounding/prompts/qwen3vl_point_grounding.md").read_text(encoding="utf-8")
    api_row = {
        "event_id": "secret_event",
        "instruction_text": "Point to the thing the user indicated.",
        "utterance_text": "that one",
        "target_description": "SECRET_TARGET_NAME",
    }
    bounds = {
        "x_q05": -10.0,
        "x_q95": 10.0,
        "y_q05": -2.0,
        "y_q95": 3.0,
        "z_q05": 0.0,
        "z_q95": 20.0,
        "robust_diagonal": 22.9,
    }
    evidence = [
        {
            "panel_id": "P1",
            "frame_path": "/tmp/frame.jpg",
            "relative_sample_time_seconds": 1.0,
            "selection_score": 2.0,
            "selection_reason": "gaze_valid",
            "telemetry": {
                "camera_position": [0.0, 1.0, 2.0],
                "camera_forward_world": [0.0, 0.0, 1.0],
                "camera_right_world": [1.0, 0.0, 0.0],
                "camera_up_world": [0.0, 1.0, 0.0],
                "camera_fov_degrees": 100.0,
                "gaze_valid": True,
                "gaze_origin": [0.0, 1.0, 2.0],
                "gaze_direction_world": [0.0, 0.0, 1.0],
                "gaze_hit": [0.0, 1.0, 8.0],
                "hand_valid": False,
                "hand_origin": None,
                "hand_direction_world": None,
                "hand_hit": None,
            },
        }
    ]
    prompt = build_prompt(template, api_row, bounds, evidence)
    assert "SECRET_TARGET_NAME" not in prompt
    assert "999.123" not in prompt
    assert "gt_anchor" not in prompt.lower()

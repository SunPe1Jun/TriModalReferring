import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from select_target_free_evidence import select_evidence_frames


def sample(timestamp, gaze_x, hand_x):
    return {
        "timestamp": timestamp,
        "eyeGaze": {
            "isEyeOpen": True,
            "gazeOrigin": {"x": 0.0, "y": 0.0, "z": 0.0},
            "gazeVector": {"x": 0.0, "y": 0.0, "z": 1.0},
            "gazePoint": {"x": gaze_x, "y": 0.0, "z": 5.0},
        },
        "handData": {
            "isRightHandTracked": True,
            "isLeftHandTracked": False,
            "rightHand": {"joints": [{"id": 8, "position": {"x": 0.0, "y": 0.0, "z": 0.0}}]},
            "rightIndexFingerRayHitPoint": {"x": hand_x, "y": 0.0, "z": 5.0},
        },
        "cameraPosition": {"x": 0.0, "y": 1.0, "z": 0.0},
        "cameraRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "cameraFOV": 100.0,
    }


def test_evidence_selection_uses_raw_cues_without_gt(tmp_path):
    json_path = tmp_path / "multimodal_data.json"
    payload = [
        sample("2025-01-01T00:00:00+00:00", 0.0, 0.0),
        sample("2025-01-01T00:00:00.500000+00:00", 0.1, 0.2),
        sample("2025-01-01T00:00:01+00:00", 0.2, 0.4),
        sample("2025-01-01T00:00:01.500000+00:00", 0.3, 0.6),
    ]
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    api_row = {
        "json_path": str(json_path),
        "video_path": "",
        "t_start": "0.0",
        "t_end": "1.5",
        "target_description": "SECRET_GT_SHOULD_NOT_BE_USED",
    }
    frames, stats = select_evidence_frames(api_row, tmp_path / "frames", no_extract_frames=True, max_frames=3)
    assert 1 <= len(frames) <= 3
    assert stats["candidate_count"] > 0
    assert all("SECRET" not in frame.selection_reason for frame in frames)

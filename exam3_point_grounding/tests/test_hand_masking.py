import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_masking import MASK_COLOR, _mask_image


def test_projected_hand_region_is_covered(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "masked.jpg"
    Image.new("RGB", (256, 256), (20, 40, 80)).save(source)
    joints = [
        {"id": 0, "position": {"x": -0.04, "y": -0.04, "z": 1.0}},
        {"id": 1, "position": {"x": 0.04, "y": 0.04, "z": 1.0}},
    ]
    sample = {
        "cameraPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cameraRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "cameraFOV": 90.0,
        "handData": {
            "isLeftHandTracked": False,
            "isRightHandTracked": True,
            "rightHand": {"joints": joints},
        },
    }
    audit = _mask_image(source, output, sample)
    masked = Image.open(output).convert("RGB")
    center = masked.getpixel((128, 128))
    assert audit["status"] == "masked"
    assert audit["mask_fraction"] > 0.1
    assert all(abs(center[index] - MASK_COLOR[index]) <= 4 for index in range(3))


def test_no_tracked_hand_keeps_zero_mask(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "masked.jpg"
    Image.new("RGB", (128, 128), (20, 40, 80)).save(source)
    sample = {
        "cameraPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cameraRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "cameraFOV": 90.0,
        "handData": {"isLeftHandTracked": False, "isRightHandTracked": False},
    }
    audit = _mask_image(source, output, sample)
    assert audit["status"] == "no_tracked_hand"
    assert audit["mask_pixels"] == 0


def test_tracked_offscreen_hand_does_not_create_inverted_box(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "masked.jpg"
    Image.new("RGB", (128, 128), (20, 40, 80)).save(source)
    sample = {
        "cameraPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cameraRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "cameraFOV": 90.0,
        "handData": {
            "isLeftHandTracked": False,
            "isRightHandTracked": True,
            "rightHand": {"joints": [{"position": {"x": 2.5, "y": 0.0, "z": 1.0}}]},
        },
    }
    audit = _mask_image(source, output, sample)
    assert audit["status"] == "tracked_offscreen"
    assert audit["mask_pixels"] == 0

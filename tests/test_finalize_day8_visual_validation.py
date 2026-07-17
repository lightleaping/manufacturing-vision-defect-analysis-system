from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.finalize_day8_visual_validation import (
    finalize_visual_validation_payload,
    read_json_object,
    validate_png_screenshot,
    write_json_atomically,
)


def _write_png(path: Path, *, size: tuple[int, int] = (120, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path, format="PNG")


def _payload() -> dict[str, object]:
    return {
        "project": "Manufacturing Vision Defect Analysis System",
        "run_name": "day8_streamlit_dashboard_validation",
        "health": {
            "status": "ok",
            "model_loaded": True,
        },
        "normal_image": {
            "response": {
                "prediction": 0,
                "prediction_class_name": "NORMAL",
            }
        },
        "defect_image": {
            "response": {
                "prediction": 1,
                "prediction_class_name": "DEFECT",
            }
        },
        "ui_visual_validation_completed": False,
        "screenshot_artifacts": [],
    }


def test_validate_png_screenshot_returns_metadata(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    _write_png(screenshot, size=(320, 240))

    metadata = validate_png_screenshot(screenshot)

    assert metadata["format"] == "PNG"
    assert metadata["width"] == 320
    assert metadata["height"] == 240
    assert int(metadata["size_bytes"]) > 0


def test_validate_png_screenshot_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_png_screenshot(tmp_path / "missing.png")


def test_validate_png_screenshot_rejects_invalid_content(tmp_path: Path) -> None:
    screenshot = tmp_path / "broken.png"
    screenshot.write_text("not png", encoding="utf-8")

    with pytest.raises(ValueError, match="readable image"):
        validate_png_screenshot(screenshot)


def test_finalize_visual_validation_payload_records_two_screenshots(
    tmp_path: Path,
) -> None:
    normal = tmp_path / "normal.png"
    defect = tmp_path / "defect.png"
    _write_png(normal)
    _write_png(defect)

    result = finalize_visual_validation_payload(
        _payload(),
        screenshot_paths=(normal, defect),
    )

    assert result["ui_visual_validation_completed"] is True
    assert result["ui_visual_validation_result"] == "PASS"
    artifacts = result["screenshot_artifacts"]
    assert isinstance(artifacts, list)
    assert [item["label"] for item in artifacts] == ["NORMAL", "DEFECT"]


def test_finalize_visual_validation_payload_rejects_unready_model(
    tmp_path: Path,
) -> None:
    normal = tmp_path / "normal.png"
    defect = tmp_path / "defect.png"
    _write_png(normal)
    _write_png(defect)
    payload = _payload()
    payload["health"]["model_loaded"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="model_loaded"):
        finalize_visual_validation_payload(
            payload,
            screenshot_paths=(normal, defect),
        )


def test_json_read_and_atomic_write_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = {"value": 1, "text": "정상"}

    write_json_atomically(path=path, payload=payload)

    assert read_json_object(path) == payload
    assert not path.with_name("artifact.json.tmp").exists()

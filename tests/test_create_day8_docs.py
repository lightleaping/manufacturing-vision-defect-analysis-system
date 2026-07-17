from __future__ import annotations

import copy
from pathlib import Path

import pytest
from PIL import Image

import scripts.create_day8_docs as docs
from scripts.create_day8_docs import (
    README_END_MARKER,
    README_START_MARKER,
    build_day8_report,
    build_readme_section,
    upsert_marked_section,
    validate_day8_payload,
)


def _response(
    *,
    prediction: int,
    class_name: str,
    probability: float,
    raw_logit: float,
    filename: str,
) -> dict[str, object]:
    return {
        "prediction": prediction,
        "prediction_class_name": class_name,
        "defect_probability": probability,
        "normal_probability": 1.0 - probability,
        "raw_logit": raw_logit,
        "classification_threshold": 0.5,
        "model_name": "ResNet18Transfer",
        "model_version": "resnet18_transfer_best",
        "positive_class": "DEFECT",
        "original_filename": filename,
        "content_type": "image/jpeg",
        "image_width": 300,
        "image_height": 300,
        "image_mode": "RGB",
        "inference_time_ms": 25.0,
    }


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 120), "white").save(path, format="PNG")


def _payload(tmp_path: Path) -> dict[str, object]:
    normal_screenshot = tmp_path / "normal.png"
    defect_screenshot = tmp_path / "defect.png"
    _write_png(normal_screenshot)
    _write_png(defect_screenshot)

    return {
        "project": "Manufacturing Vision Defect Analysis System",
        "run_name": "day8_streamlit_dashboard_validation",
        "base_url": "http://127.0.0.1:8000",
        "health": {
            "status": "ok",
            "service": "Manufacturing Vision Defect Analysis System",
            "model_loaded": True,
            "model_name": "ResNet18Transfer",
            "device": "cpu",
        },
        "normal_image": {
            "path": "data/test/ok.jpeg",
            "response": _response(
                prediction=0,
                class_name="NORMAL",
                probability=0.013476800173521042,
                raw_logit=-4.293217182159424,
                filename="cast_ok_0_7631.jpeg",
            ),
        },
        "defect_image": {
            "path": "data/test/def.jpeg",
            "response": _response(
                prediction=1,
                class_name="DEFECT",
                probability=0.999903678894043,
                raw_logit=9.247676849365234,
                filename="cast_def_0_1414.jpeg",
            ),
        },
        "runtime_seconds": 3.62,
        "ui_visual_validation_completed": True,
        "ui_visual_validation_result": "PASS",
        "screenshot_artifacts": [
            {
                "label": "NORMAL",
                "path": str(normal_screenshot),
                "format": "PNG",
                "width": 200,
                "height": 120,
                "size_bytes": normal_screenshot.stat().st_size,
            },
            {
                "label": "DEFECT",
                "path": str(defect_screenshot),
                "format": "PNG",
                "width": 200,
                "height": 120,
                "size_bytes": defect_screenshot.stat().st_size,
            },
        ],
    }


def test_validate_day8_payload_accepts_completed_validation(tmp_path: Path) -> None:
    validate_day8_payload(_payload(tmp_path))


def test_validate_day8_payload_rejects_incomplete_visual_check(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    payload["ui_visual_validation_completed"] = False

    with pytest.raises(ValueError, match="ui_visual_validation_completed"):
        validate_day8_payload(payload)


def test_validate_day8_payload_rejects_missing_screenshot(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    screenshot = payload["screenshot_artifacts"][0]  # type: ignore[index]
    Path(screenshot["path"]).unlink()  # type: ignore[arg-type,index]

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_day8_payload(payload)


def test_build_day8_report_contains_actual_results(tmp_path: Path) -> None:
    report = build_day8_report(
        _payload(tmp_path),
        regression_test_count=1315,
        warning_count=1,
    )

    assert "Day 8 — Streamlit Image Inference Dashboard" in report
    assert "0.013476800174" in report
    assert "0.999903678894" in report
    assert "1315 passed" in report
    assert "Streamlit" in report


def test_build_readme_section_contains_single_marker_pair(tmp_path: Path) -> None:
    section = build_readme_section(
        _payload(tmp_path),
        regression_test_count=1315,
        warning_count=1,
    )

    assert section.count(README_START_MARKER) == 1
    assert section.count(README_END_MARKER) == 1
    assert "Full Regression Tests        : 1315 passed" in section
    assert "UI Visual Validation: PASS" in section


def test_upsert_marked_section_appends_and_replaces(tmp_path: Path) -> None:
    section = build_readme_section(
        _payload(tmp_path),
        regression_test_count=1315,
        warning_count=0,
    )
    appended = upsert_marked_section(
        original_text="# Project\n",
        section_text=section,
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )
    replaced = upsert_marked_section(
        original_text=appended,
        section_text=section.replace("1315", "1316"),
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )

    assert replaced.count(README_START_MARKER) == 1
    assert replaced.count(README_END_MARKER) == 1
    assert "1316 passed" in replaced
    assert "1315 passed" not in replaced


def test_upsert_marked_section_rejects_partial_marker() -> None:
    with pytest.raises(ValueError, match="only one"):
        upsert_marked_section(
            original_text=f"# Project\n{README_START_MARKER}\n",
            section_text="section",
            start_marker=README_START_MARKER,
            end_marker=README_END_MARKER,
        )


def test_document_builders_do_not_mutate_payload(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    original = copy.deepcopy(payload)

    _ = build_day8_report(
        payload,
        regression_test_count=1315,
        warning_count=1,
    )
    _ = build_readme_section(
        payload,
        regression_test_count=1315,
        warning_count=1,
    )

    assert payload == original


def test_builders_reject_invalid_test_counts(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        build_day8_report(
            payload,
            regression_test_count=0,
            warning_count=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        build_readme_section(
            payload,
            regression_test_count=1,
            warning_count=-1,
        )

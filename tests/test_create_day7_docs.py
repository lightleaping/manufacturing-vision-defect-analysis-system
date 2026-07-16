from __future__ import annotations

import copy

import pytest

from scripts.create_day7_docs import (
    README_END_MARKER,
    README_START_MARKER,
    build_day7_report,
    build_readme_section,
    update_regression_test_count,
    upsert_marked_section,
    validate_validation_payload,
)


def _valid_response(
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
        "inference_time_ms": 50.0,
    }


def _valid_payload() -> dict[str, object]:
    return {
        "project": "Manufacturing Vision Defect Analysis System",
        "run_name": "day7_fastapi_inference_validation",
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
            "response": _valid_response(
                prediction=0,
                class_name="NORMAL",
                probability=0.013476800173521042,
                raw_logit=-4.293217182159424,
                filename="ok.jpeg",
            ),
        },
        "defect_image": {
            "path": "data/test/def.jpeg",
            "response": _valid_response(
                prediction=1,
                class_name="DEFECT",
                probability=0.999903678894043,
                raw_logit=9.247676849365234,
                filename="def.jpeg",
            ),
        },
        "runtime_seconds": 14.21,
    }


def test_validate_validation_payload_accepts_realistic_payload() -> None:
    validate_validation_payload(_valid_payload())


def test_validate_validation_payload_rejects_unloaded_model() -> None:
    payload = _valid_payload()
    payload["health"]["model_loaded"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="model_loaded"):
        validate_validation_payload(payload)


def test_validate_validation_payload_rejects_wrong_class() -> None:
    payload = _valid_payload()
    payload["normal_image"]["response"]["prediction_class_name"] = "DEFECT"  # type: ignore[index]

    with pytest.raises(ValueError, match="prediction_class_name"):
        validate_validation_payload(payload)


def test_validate_validation_payload_rejects_probability_sum_error() -> None:
    payload = _valid_payload()
    response = payload["defect_image"]["response"]  # type: ignore[index]
    response["normal_probability"] = 0.5  # type: ignore[index]

    with pytest.raises(ValueError, match="sum to 1"):
        validate_validation_payload(payload)


def test_build_day7_report_contains_actual_results() -> None:
    report = build_day7_report(_valid_payload())

    assert "Day 7 — FastAPI Image Inference API" in report
    assert "/api/v1/predictions" in report
    assert "0.013476800174" in report
    assert "0.999903678894" in report
    assert "1255 passed" in report


def test_build_readme_section_contains_single_marker_pair() -> None:
    section = build_readme_section(_valid_payload())

    assert section.count(README_START_MARKER) == 1
    assert section.count(README_END_MARKER) == 1
    assert "Day 7 API Tests      : 40 passed" in section
    assert "Full Regression Tests: 1255 passed" in section


def test_upsert_marked_section_appends_when_markers_are_missing() -> None:
    result = upsert_marked_section(
        original_text="# Project\n",
        section_text=build_readme_section(_valid_payload()),
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )

    assert result.startswith("# Project")
    assert result.count(README_START_MARKER) == 1
    assert result.count(README_END_MARKER) == 1


def test_upsert_marked_section_replaces_existing_section_without_duplicates() -> None:
    old_text = (
        "# Project\n\n"
        f"{README_START_MARKER}\nold\n{README_END_MARKER}\n\n"
        "## Next\n"
    )

    first_result = upsert_marked_section(
        original_text=old_text,
        section_text=build_readme_section(_valid_payload()),
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )
    second_result = upsert_marked_section(
        original_text=first_result,
        section_text=build_readme_section(_valid_payload()),
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )

    assert second_result.count(README_START_MARKER) == 1
    assert second_result.count(README_END_MARKER) == 1
    assert "\nold\n" not in second_result
    assert "## Next" in second_result


def test_upsert_marked_section_rejects_partial_marker() -> None:
    with pytest.raises(ValueError, match="only one"):
        upsert_marked_section(
            original_text=f"# Project\n{README_START_MARKER}\n",
            section_text=build_readme_section(_valid_payload()),
            start_marker=README_START_MARKER,
            end_marker=README_END_MARKER,
        )


def test_update_regression_test_count_replaces_old_value() -> None:
    original = "Full Regression Tests : 1204 passed"

    result = update_regression_test_count(original)

    assert result == "Full Regression Tests : 1255 passed"


def test_document_builders_do_not_mutate_payload() -> None:
    payload = _valid_payload()
    original = copy.deepcopy(payload)

    _ = build_day7_report(payload)
    _ = build_readme_section(payload)

    assert payload == original

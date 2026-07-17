"""Tests for Day 14 final documentation creation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_day14_docs import (
    BACKUP_PATH,
    CLASSIFICATION_CHECKPOINT,
    CONTEXT_REBUILD_PATH,
    DAY13_SUMMARY_PATH,
    DAY14_EVIDENCE_PATH,
    DAY14_INSPECTION_PATH,
    DETECTION_CHECKPOINT,
    DETECTION_EVALUATION_PATH,
    DETECTION_FAILURE_PATH,
    DETECTION_SPLIT_MANIFEST,
    CLASSIFICATION_EVALUATION_PATH,
    Day14DocumentationError,
    README_END_MARKER,
    README_START_MARKER,
    REPORT_PATH,
    SUMMARY_PATH,
    ValidationResult,
    build_architecture_mermaid,
    build_readme_section,
    build_report,
    create_day14_documents,
    update_marker_block,
    _contains_numeric,
    _normalized_numeric_candidates,
    validate_expected_numeric_evidence,
    validate_generated_text,
    validate_repository_evidence,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _build_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"

    _write(
        root / "README.md",
        "# Existing Project\n\nExisting content.\n",
    )
    _write(root / CLASSIFICATION_CHECKPOINT, "classification")
    _write(root / DETECTION_CHECKPOINT, "detection")
    _write_json(root / DETECTION_SPLIT_MANIFEST, {"status": "ok"})

    _write_json(
        root / CLASSIFICATION_EVALUATION_PATH,
        {
            "metrics": {
                "accuracy": 0.9734,
                "precision": 0.9717,
                "recall": 0.9868,
                "f1": 0.9792,
            },
            "confusion_matrix": {
                "tn": 249,
                "fp": 13,
                "fn": 6,
                "tp": 447,
            },
        },
    )
    _write_json(
        root / DETECTION_EVALUATION_PATH,
        {
            "global": {
                "tp": 226,
                "fp": 52,
                "fn": 203,
                "precision": 0.812950,
                "recall": 0.526807,
                "f1": 0.639321,
                "mean_matched_iou": 0.752338,
                "map_50": 0.707726,
                "project_ap_50_95": 0.310533,
            },
            "per_class": {
                "patches": {
                    "f1": 0.841026,
                    "ap": 0.888495,
                },
                "crazing": {
                    "recall": 0.025316,
                    "f1": 0.048780,
                    "ap": 0.522723,
                },
            },
        },
    )
    _write_json(
        root / DETECTION_FAILURE_PATH,
        {
            "test_images": 182,
            "images_with_failures": 129,
            "failure_events": 229,
            "breakdown": {
                "low_confidence_correct": 140,
                "false_negative": 37,
                "low_iou": 25,
                "false_positive": 23,
                "duplicate": 3,
                "wrong_class": 1,
            },
        },
    )
    _write_json(
        root / DAY13_SUMMARY_PATH,
        {
            "tests": {
                "targeted_test_count": 92,
                "regression_test_count": 1668,
                "warning_count": 1,
            },
            "evidence": {
                "dashboard": {
                    "checkpoint_epoch": 3,
                },
            },
            "manual_browser_check_status": "not_recorded",
        },
    )
    _write_json(
        root / DAY14_EVIDENCE_PATH,
        {
            "schema_version": 2,
            "status": {"overall": "PASS"},
            "fastapi": {
                "expected_status": {
                    "GET /api/v1/health": True,
                    "POST /api/v1/predictions": True,
                    "POST /api/v1/detection/predictions": True,
                }
            },
            "important_paths": {
                "classification_checkpoint": {
                    "found_any": True,
                    "found": [
                        "models/checkpoints/resnet18_transfer_best.pt"
                    ],
                },
                "detection_checkpoint_best": {
                    "found_any": True,
                    "found": [
                        "models/detection/day12_detection_best.pt"
                    ],
                },
                "detection_split_manifest": {
                    "found_any": True,
                    "found": [
                        "data/processed/neu_det/splits.json"
                    ],
                },
            },
        },
    )
    _write_json(
        root / DAY14_INSPECTION_PATH,
        {
            "schema_version": 4,
            "status": {"overall": "PASS"},
        },
    )
    _write_json(
        root / CONTEXT_REBUILD_PATH,
        {
            "status": "PASS",
            "rebuilt": {
                "mojibake_tokens": [],
            },
        },
    )
    return root


def _test_result() -> ValidationResult:
    return ValidationResult(
        targeted_count=32,
        regression_count=1700,
        warning_count=1,
        runtime_seconds=100.25,
    )


def test_numeric_evidence_validation_passes(tmp_path: Path) -> None:
    root = _build_project(tmp_path)

    result = validate_expected_numeric_evidence(root=root)

    assert result["status"] == "PASS"
    assert result["artifact_count"] == 4


def test_numeric_evidence_validation_rejects_mismatch(
    tmp_path: Path,
) -> None:
    root = _build_project(tmp_path)
    _write_json(
        root / CLASSIFICATION_EVALUATION_PATH,
        {"accuracy": 0.5},
    )

    with pytest.raises(Day14DocumentationError, match="일치하지 않습니다"):
        validate_expected_numeric_evidence(root=root)


def test_repository_evidence_validation_passes(tmp_path: Path) -> None:
    root = _build_project(tmp_path)

    result = validate_repository_evidence(root)

    assert result["status"] == "PASS"
    assert result["manual_browser_check_status"] == "not_recorded"
    assert all(result["expected_endpoints"].values())


def test_repository_evidence_requires_pass_status(
    tmp_path: Path,
) -> None:
    root = _build_project(tmp_path)
    _write_json(
        root / DAY14_INSPECTION_PATH,
        {"status": {"overall": "WARN"}},
    )

    with pytest.raises(Day14DocumentationError, match="PASS"):
        validate_repository_evidence(root)


def test_marker_upsert_appends_once() -> None:
    original = "# Existing\n"
    section = (
        f"{README_START_MARKER}\n"
        "## Day 14\n"
        f"{README_END_MARKER}"
    )

    updated = update_marker_block(original, section)

    assert updated.count(README_START_MARKER) == 1
    assert updated.count(README_END_MARKER) == 1
    assert updated.startswith("# Existing")


def test_marker_upsert_replaces_idempotently() -> None:
    first = update_marker_block(
        "# Existing\n",
        f"{README_START_MARKER}\nFirst\n{README_END_MARKER}",
    )
    second = update_marker_block(
        first,
        f"{README_START_MARKER}\nSecond\n{README_END_MARKER}",
    )

    assert "First" not in second
    assert "Second" in second
    assert second.count(README_START_MARKER) == 1


def test_marker_upsert_rejects_unbalanced_markers() -> None:
    with pytest.raises(Day14DocumentationError, match="Marker 수"):
        update_marker_block(
            README_START_MARKER,
            f"{README_START_MARKER}\nX\n{README_END_MARKER}",
        )


def test_architecture_documents_three_pipeline_roles() -> None:
    diagram = build_architecture_mermaid()

    assert "Classification" in diagram
    assert "Object Detection" in diagram
    assert "OpenCV Auxiliary Analysis" in diagram
    assert "Ground Truth나 Detection Prediction이 아닌" in diagram


def test_readme_section_contains_truthful_boundaries() -> None:
    section = build_readme_section(
        test_result=_test_result(),
        manual_browser_status="not_recorded",
    )

    assert "not_recorded" in section
    assert "공식 COCOeval 결과로 표현하지 않습니다" in section
    assert "32 passed" in section
    assert "1700 passed" in section


def test_report_contains_fixed_project_name_and_portfolio_sections() -> None:
    report = build_report(
        test_result=_test_result(),
        manual_browser_status="not_recorded",
    )

    assert "Manufacturing Vision Defect Analysis System" in report
    assert "## 13. Portfolio Summary" in report
    assert "## 14. Interview Guide" in report
    assert "문제 상황" in report
    assert "해결 방안과 고민" in report
    assert "적용" in report
    assert "효과와 의미" in report


def test_generated_text_has_no_forbidden_overclaim() -> None:
    report = build_report(
        test_result=_test_result(),
        manual_browser_status="not_recorded",
    )

    validate_generated_text(report)


def test_generated_text_rejects_forbidden_overclaim() -> None:
    with pytest.raises(Day14DocumentationError, match="금지된"):
        validate_generated_text(
            "실제 생산 환경에서 검증 완료"
        )


def test_create_documents_writes_report_summary_readme_and_backup(
    tmp_path: Path,
) -> None:
    root = _build_project(tmp_path)
    original_readme = (root / "README.md").read_bytes()

    report_path, summary_path, readme_path = create_day14_documents(
        project_root=root,
        test_result=_test_result(),
    )

    assert report_path == root / REPORT_PATH
    assert summary_path == root / SUMMARY_PATH
    assert readme_path == root / "README.md"
    assert report_path.is_file()
    assert summary_path.is_file()
    assert (root / BACKUP_PATH).read_bytes() == original_readme

    readme = readme_path.read_text(encoding="utf-8")
    assert readme.count(README_START_MARKER) == 1
    assert readme.count(README_END_MARKER) == 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["tests"]["regression_test_count"] == 1700
    assert summary["manual_browser_check_status"] == "not_recorded"
    assert summary["write_policy"]["application_source_modified"] is False


def test_create_documents_is_idempotent(tmp_path: Path) -> None:
    root = _build_project(tmp_path)

    create_day14_documents(
        project_root=root,
        test_result=_test_result(),
    )
    first = (root / "README.md").read_text(encoding="utf-8")

    create_day14_documents(
        project_root=root,
        test_result=_test_result(),
    )
    second = (root / "README.md").read_text(encoding="utf-8")

    assert first == second
    assert second.count(README_START_MARKER) == 1


def test_create_documents_rejects_non_recorded_browser_change(
    tmp_path: Path,
) -> None:
    root = _build_project(tmp_path)
    _write_json(
        root / DAY13_SUMMARY_PATH,
        {
            "targeted_test_count": 92,
            "regression_test_count": 1668,
            "warning_count": 1,
            "checkpoint_epoch": 3,
            "manual_browser_check_status": "passed",
        },
    )

    with pytest.raises(Day14DocumentationError, match="수동 Browser"):
        create_day14_documents(
            project_root=root,
            test_result=_test_result(),
        )



def test_numeric_candidates_normalize_percentage_to_ratio() -> None:
    candidates = _normalized_numeric_candidates(
        97.342657342657,
        0.9734,
    )

    assert candidates == (
        97.342657342657,
        0.97342657342657,
    )


def test_contains_numeric_accepts_precise_ratio_for_rounded_readme_value() -> None:
    assert _contains_numeric(
        [0.9734265734265735],
        0.9734,
    )


def test_contains_numeric_accepts_percentage_for_ratio_readme_value() -> None:
    assert _contains_numeric(
        [97.34265734265735],
        0.9734,
    )


def test_contains_numeric_rejects_unrelated_metric() -> None:
    assert not _contains_numeric(
        [97.50],
        0.9734,
    )


def test_classification_percentage_artifact_passes_numeric_validation(
    tmp_path: Path,
) -> None:
    root = _build_project(tmp_path)
    _write_json(
        root / CLASSIFICATION_EVALUATION_PATH,
        {
            "metrics_percent": {
                "accuracy": 97.34265734265735,
                "precision": 97.17391304347827,
                "recall": 98.67549668874172,
                "f1": 97.91894852135816,
            },
            "confusion_matrix": {
                "tn": 249,
                "fp": 13,
                "fn": 6,
                "tp": 447,
            },
        },
    )

    result = validate_expected_numeric_evidence(root=root)

    assert result["status"] == "PASS"

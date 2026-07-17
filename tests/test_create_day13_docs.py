from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_day13_docs import (
    API_INSPECTION_ARTIFACT,
    API_SMOKE_ARTIFACT,
    BACKUP_ROOT,
    BEST_CHECKPOINT,
    DASHBOARD_INSPECTION_ARTIFACT,
    DASHBOARD_OVERLAY_FIGURE,
    DASHBOARD_VALIDATION_ARTIFACT,
    PREREQUISITES_ARTIFACT,
    README_END,
    README_START,
    REPORT_PATH,
    SUMMARY_ARTIFACT,
    collect_day13_evidence,
    create_day13_docs,
    render_day13_report,
    update_marker_block,
)


def write_json(
    root: Path,
    relative_path: Path,
    payload: dict,
) -> None:
    path = root / relative_path
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_required_files(root: Path) -> None:
    implementation_files = (
        "src/api/detection_config.py",
        "src/api/detection_inference_service.py",
        "src/api/detection_model_loader.py",
        "src/dashboard/detection_api_client.py",
        "src/dashboard/detection_session_state.py",
        "src/dashboard/detection_ui_helpers.py",
        "src/dashboard/detection_page.py",
        "src/dashboard/pages/2_Detection.py",
    )
    for relative_path in implementation_files:
        path = root / relative_path
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "# file\n",
            encoding="utf-8",
        )

    checkpoint = root / BEST_CHECKPOINT
    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint.write_bytes(b"checkpoint")

    figure = root / DASHBOARD_OVERLAY_FIGURE
    figure.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.write_bytes(b"png")


def create_artifacts(
    root: Path,
    *,
    smoke_passed: bool = True,
    dashboard_passed: bool = True,
) -> None:
    write_json(
        root,
        PREREQUISITES_ARTIFACT,
        {
            "validation": {
                "all_passed": True,
            }
        },
    )
    write_json(
        root,
        API_INSPECTION_ARTIFACT,
        {
            "validation_passed": True,
            "checkpoint": {
                "epoch_number": 3,
                "best_metric": 0.677418,
            },
            "api_policy": {
                "endpoint": "/api/v1/detection/predictions",
                "default_score_threshold": 0.5,
                "minimum_score_threshold": 0.05,
                "maximum_score_threshold": 0.95,
                "iou_threshold": 0.5,
            },
        },
    )
    write_json(
        root,
        API_SMOKE_ARTIFACT,
        {
            "validation_passed": smoke_passed,
            "request": {
                "endpoint": "/api/v1/detection/predictions",
                "image_path": "sample.jpg",
                "score_threshold": 0.5,
            },
            "response": {
                "status_code": 200,
                "payload": {
                    "checkpoint_epoch": 3,
                    "architecture": (
                        "fasterrcnn_mobilenet_v3_large_320_fpn"
                    ),
                    "detection_count": 1,
                    "detections": [
                        {
                            "label_id": 3,
                        }
                    ],
                    "inference_time_ms": 100.0,
                },
            },
        },
    )
    write_json(
        root,
        DASHBOARD_INSPECTION_ARTIFACT,
        {
            "validation_passed": True,
            "forbidden_imports": [],
            "checks": {
                "required_files_exist": True,
                "detection_endpoint_present": True,
                "default_threshold_is_0_5": True,
                "threshold_range_present": True,
                "api_client_only": True,
                "prediction_overlay_present": True,
                "prediction_table_present": True,
                "opencv_distinction_present": True,
                "ground_truth_warning_present": True,
            },
        },
    )
    write_json(
        root,
        DASHBOARD_VALIDATION_ARTIFACT,
        {
            "validation_passed": dashboard_passed,
            "endpoint": "/api/v1/detection/predictions",
            "image_path": "sample.jpg",
            "score_threshold": 0.5,
            "detection_count": 1,
            "inference_time_ms": 120.0,
            "checkpoint_epoch": 3,
            "checkpoint_metric_value": 0.677418,
        },
    )


def prepared_root(
    tmp_path: Path,
) -> Path:
    create_required_files(tmp_path)
    create_artifacts(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Project\n",
        encoding="utf-8",
    )
    return tmp_path


def test_update_marker_block_appends_once() -> None:
    updated = update_marker_block(
        "# Project\n",
        f"{README_START}\nDay 13\n{README_END}",
    )

    assert updated.count(README_START) == 1
    assert updated.count(README_END) == 1
    assert "Day 13" in updated


def test_update_marker_block_replaces_existing_block() -> None:
    original = (
        "# Project\n\n"
        f"{README_START}\nold\n{README_END}\n"
    )
    updated = update_marker_block(
        original,
        f"{README_START}\nnew\n{README_END}",
    )

    assert "old" not in updated
    assert "new" in updated
    assert updated.count(README_START) == 1
    assert updated.count(README_END) == 1


def test_update_marker_block_rejects_orphan_marker() -> None:
    with pytest.raises(
        ValueError,
        match="exactly one pair",
    ):
        update_marker_block(
            f"# Project\n{README_START}\n",
            "section",
        )


def test_collect_day13_evidence_accepts_valid_artifacts(
    tmp_path: Path,
) -> None:
    root = prepared_root(tmp_path)

    evidence = collect_day13_evidence(
        project_root=root
    )

    assert evidence["checkpoint"]["epoch"] == 3
    assert (
        evidence["api"]["endpoint"]
        == "/api/v1/detection/predictions"
    )
    assert evidence["dashboard"]["api_client_only"] is True


def test_collect_day13_evidence_rejects_failed_smoke(
    tmp_path: Path,
) -> None:
    create_required_files(tmp_path)
    create_artifacts(
        tmp_path,
        smoke_passed=False,
    )

    with pytest.raises(
        ValueError,
        match="Smoke Test",
    ):
        collect_day13_evidence(
            project_root=tmp_path
        )


def test_collect_day13_evidence_requires_overlay(
    tmp_path: Path,
) -> None:
    root = prepared_root(tmp_path)
    (root / DASHBOARD_OVERLAY_FIGURE).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Overlay Figure",
    ):
        collect_day13_evidence(
            project_root=root
        )


def test_render_report_does_not_claim_day14_completion(
    tmp_path: Path,
) -> None:
    root = prepared_root(tmp_path)
    evidence = collect_day13_evidence(
        project_root=root
    )

    report = render_day13_report(
        evidence=evidence,
        targeted_test_count=92,
        regression_test_count=1668,
        warning_count=1,
        manual_browser_check_status="not_recorded",
    )

    assert "Day 14 범위로 남긴다" in report
    assert "Day 14에서는 다음을 진행한다" in report
    assert "브라우저 수동 시각 검증 결과는" in report
    assert "완료했다고 표현하지 않는다" in report


def test_create_day13_docs_writes_outputs_and_backup(
    tmp_path: Path,
) -> None:
    root = prepared_root(tmp_path)

    report, summary, readme = create_day13_docs(
        project_root=root,
        targeted_test_count=92,
        regression_test_count=1668,
        warning_count=1,
    )

    assert report == root / REPORT_PATH
    assert summary == root / SUMMARY_ARTIFACT
    assert readme == root / "README.md"
    assert report.is_file()
    assert summary.is_file()
    assert readme.read_text(
        encoding="utf-8"
    ).count(README_START) == 1
    assert (
        root
        / BACKUP_ROOT
        / "README.md.before_day13_docs"
    ).is_file()


def test_create_day13_docs_is_idempotent(
    tmp_path: Path,
) -> None:
    root = prepared_root(tmp_path)

    for _ in range(2):
        create_day13_docs(
            project_root=root,
            targeted_test_count=92,
            regression_test_count=1668,
            warning_count=1,
        )

    readme_text = (
        root
        / "README.md"
    ).read_text(encoding="utf-8")
    assert readme_text.count(README_START) == 1
    assert readme_text.count(README_END) == 1


def test_summary_records_manual_browser_status(
    tmp_path: Path,
) -> None:
    root = prepared_root(tmp_path)

    _, summary_path, _ = create_day13_docs(
        project_root=root,
        targeted_test_count=92,
        regression_test_count=1668,
        warning_count=1,
        manual_browser_check_status="pass",
    )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    completion = summary["completion_scope"]

    assert completion["manual_browser_check_status"] == "pass"
    assert completion["day14_final_integration_completed"] is False
    assert completion["portfolio_completed"] is False

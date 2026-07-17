"""Tests for the Day 14 final-integration prerequisite inspector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inspect_day14_final_integration_prerequisites import (
    inspect_endpoints,
    inspect_markers,
    inspect_project,
    inspect_readme_links,
    inspect_readme_test_counts,
    write_json,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_minimal_project(root: Path) -> Path:
    """실제 저장소를 건드리지 않는 최소 테스트용 프로젝트를 만든다."""

    markers = []
    for day in range(1, 14):
        markers.extend(
            [
                f"<!-- DAY{day}_SECTION_START -->",
                f"Day {day}",
                f"<!-- DAY{day}_SECTION_END -->",
            ]
        )

    _write(
        root / "README.md",
        "\n".join(markers)
        + "\n\n1668 passed, 1 warning in 96.76 seconds\n"
        + "![overlay](reports/figures/day13_detection_dashboard_overlay.png)\n",
    )
    _write(root / "requirements.txt", "pytest\n")
    _write(root / ".gitignore", ".venv/\n")

    _write(
        root / "src/api/app.py",
        """
from fastapi import FastAPI

API_PREFIX = "/api/v1"
service_api = FastAPI()

@service_api.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok"}

@service_api.post(API_PREFIX + "/predictions")
def classify():
    return {}

@service_api.post("/api/v1/detection/predictions")
def detect():
    return {}
""".strip()
        + "\n",
    )
    _write(root / "src/dashboard/app.py", "import streamlit as st\n")
    _write(root / "src/dashboard/detection_api_client.py", "class DetectionApiClient:\n    pass\n")
    _write(root / "src/dashboard/detection_session_state.py", "STATE_KEY = 'detection'\n")
    _write(root / "src/dashboard/detection_ui_helpers.py", "def draw_overlay():\n    return None\n")
    _write(root / "src/dashboard/detection_page.py", "from .detection_api_client import DetectionApiClient\n")
    _write(root / "src/dashboard/pages/2_Detection.py", "from src.dashboard.detection_page import DetectionApiClient\n")
    _write(root / "src/api/services/classification_service.py", "class ClassificationService:\n    pass\n")
    _write(root / "src/api/services/detection_service.py", "class DetectionService:\n    pass\n")
    _write(root / "src/api/schemas.py", "from dataclasses import dataclass\n")
    _write(root / "src/opencv/pipeline.py", "def analyze():\n    return {}\n")
    _write(root / "src/detection/model_factory.py", "def build_detection_model():\n    return None\n")

    _write(root / "scripts/inspect_day14_final_integration_prerequisites.py", "# placeholder\n")
    _write(root / "tests/test_placeholder.py", "def test_placeholder():\n    assert True\n")

    for day in range(1, 14):
        _write(root / "reports" / f"day{day}_summary.md", f"# Day {day}\n")

    artifact_names = (
        "day13_detection_api_stage1_inspection.json",
        "day13_detection_api_smoke_test.json",
        "day13_detection_dashboard_stage2_inspection.json",
        "day13_detection_dashboard_api_client_validation.json",
        "day13_detection_integration_summary.json",
        "day12_detection_test_evaluation.json",
        "day12_detection_failure_analysis.json",
    )
    for artifact_name in artifact_names:
        _write(
            root / "reports" / "artifacts" / artifact_name,
            json.dumps(
                {
                    "test_count": 1668,
                    "warning_count": 1,
                    "manual_browser_check_status": "not_recorded",
                }
            ),
        )

    _write(
        root / "reports" / "day13_detection_fastapi_streamlit_integration_summary.md",
        "# Day 13\n",
    )
    _write(
        root / "reports" / "figures" / "day13_detection_dashboard_overlay.png",
        "placeholder",
    )
    _write(root / "models" / "detection" / "day12_detection_latest.pt", "placeholder")
    _write(root / "models" / "detection" / "day12_detection_best.pt", "placeholder")
    _write(root / "data" / "processed" / "neu_det" / "splits.json", "{}\n")

    return root


def test_inspect_markers_accepts_one_pair_per_day() -> None:
    text = "\n".join(
        f"<!-- DAY{day}_SECTION_{kind} -->"
        for day in range(1, 14)
        for kind in ("START", "END")
    )

    result = inspect_markers(text)

    assert result["missing_days_1_to_13"] == []
    assert result["invalid_or_duplicate_pairs"] == []
    assert result["all_day_1_to_13_present"] is True
    assert result["all_pairs_single"] is True


def test_inspect_markers_detects_duplicate_start() -> None:
    text = "\n".join(
        [
            "<!-- DAY13_SECTION_START -->",
            "<!-- DAY13_SECTION_START -->",
            "<!-- DAY13_SECTION_END -->",
        ]
    )

    result = inspect_markers(text)

    assert result["invalid_or_duplicate_pairs"] == [
        {
            "marker": "DAY13_SECTION",
            "start_count": 2,
            "end_count": 1,
        }
    ]


def test_endpoint_inspection_finds_expected_routes(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)

    result = inspect_endpoints(project)

    assert result["missing_expected_endpoints"] == []
    assert len(result["endpoints"]) == 3


def test_readme_link_inspection_detects_missing_target(tmp_path: Path) -> None:
    readme = "\n".join(
        [
            "[exists](reports/example.md)",
            "[missing](reports/missing.md)",
            "[external](https://example.com)",
        ]
    )
    _write(tmp_path / "reports" / "example.md", "ok")

    result = inspect_readme_links(tmp_path, readme)

    assert result["broken_count"] == 1
    assert result["broken_relative_links"] == [{"target": "reports/missing.md"}]


def test_readme_test_count_inspection_finds_day13_count() -> None:
    result = inspect_readme_test_counts(
        "92 passed\n1668 passed, 1 warning in 96.76 seconds\n"
    )

    assert result["counts_found"] == [92, 1668]
    assert result["maximum_count_found"] == 1668
    assert result["contains_day13_final_count_1668"] is True


def test_project_inspection_does_not_modify_readme(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)
    readme_path = project / "README.md"
    before = readme_path.read_bytes()

    result = inspect_project(project)

    after = readme_path.read_bytes()
    assert before == after
    assert result["write_policy"]["readme_modified"] is False
    assert result["write_policy"]["source_modified"] is False
    assert result["write_policy"]["inspection_only"] is True


def test_project_inspection_records_manual_browser_status(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)

    result = inspect_project(project)

    assert result["manual_browser_check"]["day13_status"] == "not_recorded"
    assert "수동 브라우저 확인 전에는 완료로 변경하지 않는다" in (
        result["manual_browser_check"]["day14_policy"]
    )


def test_project_inspection_passes_dashboard_boundary(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)

    result = inspect_project(project)

    assert result["dashboard_boundary"]["api_client_only_static_check_passed"] is True
    assert result["dashboard_boundary"]["direct_model_access_findings"] == []


def test_project_inspection_detects_direct_torch_import(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project / "src/dashboard/detection_page.py",
        "import torch\n",
    )

    result = inspect_project(project)

    assert result["dashboard_boundary"]["api_client_only_static_check_passed"] is False
    assert result["status"]["overall"] == "FAIL"


def test_project_inspection_collects_day_reports(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)

    result = inspect_project(project)

    assert result["inventory"]["missing_day_reports_1_to_13"] == []
    assert result["inventory"]["counts"]["report_markdown"] >= 13


def test_project_inspection_collects_artifact_values(tmp_path: Path) -> None:
    project = _build_minimal_project(tmp_path)

    result = inspect_project(project)

    assert result["key_artifacts"]["artifact_directory_exists"] is True
    assert result["key_artifacts"]["files"]
    assert all(row["valid_json"] for row in result["key_artifacts"]["files"])


def test_write_json_creates_utf8_artifact(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "artifacts" / "inspection.json"
    payload = {"message": "한글 저장 확인"}

    write_json(payload, output)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload

def test_endpoint_inspection_supports_arbitrary_decorator_name_and_constants(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)

    result = inspect_endpoints(project)

    assert result["inspection_method"] == "python_ast"
    assert result["missing_expected_endpoints"] == []
    assert {
        (row["method"], row["path"])
        for row in result["endpoints"]
    } == {
        ("GET", "/api/v1/health"),
        ("POST", "/api/v1/predictions"),
        ("POST", "/api/v1/detection/predictions"),
    }


def test_missing_legacy_markers_are_information_not_warning(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    readme_path = project / "README.md"
    text = readme_path.read_text(encoding="utf-8")

    for day in (1, 2, 3):
        text = text.replace(f"<!-- DAY{day}_SECTION_START -->\n", "")
        text = text.replace(f"<!-- DAY{day}_SECTION_END -->\n", "")

    readme_path.write_text(text, encoding="utf-8")
    result = inspect_project(project)

    assert result["readme"]["markers"]["missing_days_1_to_13"] == [1, 2, 3]
    assert result["status"]["warning_count"] == 0
    assert any(
        "Day 1" in message and "Day 3" in message
        for message in result["status"]["information"]
    )


def test_diagnostic_context_mojibake_is_separated_from_user_facing_warning(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project / "reports/artifacts/day13_api_core_context.txt",
        "吏媛 쒖湲怨\n",
    )

    result = inspect_project(project)

    assert result["text_quality"]["finding_count"] == 0
    diagnostic = result["text_quality"]["diagnostic_artifacts"]
    assert diagnostic["file_count"] == 1
    assert diagnostic["files_with_possible_mojibake"][0]["file"] == (
        "reports/artifacts/day13_api_core_context.txt"
    )
    assert result["status"]["warning_count"] == 0



def test_inspector_excludes_its_own_detection_pattern_literals(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project / "scripts/inspect_day14_final_integration_prerequisites.py",
        "\n".join(
            [
                'PATTERNS = (',
                '    "실제 생산 환경에서 검증 완료",',
                '    "COCO 공식 mAP와 완전히 동일",',
                '    "OpenCV Contour가 실제 결함 위치",',
                '    "Detection Prediction이 Ground Truth",',
                '    "媛",',
                ')',
            ]
        )
        + "\n",
    )

    result = inspect_project(project)

    assert result["schema_version"] == 5
    assert result["text_quality"]["finding_count"] == 0
    assert result["text_quality"]["excluded_self_referential_files"] == [
        "scripts/inspect_day14_final_integration_prerequisites.py",
        "scripts/rebuild_day13_api_core_context.py",
        "scripts/create_day14_docs.py",
    ]
    assert result["status"]["warning_count"] == 0


def test_user_facing_overclaim_is_still_detected_after_self_exclusion(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    readme_path = project / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8")
        + "\n실제 생산 환경에서 검증 완료\n",
        encoding="utf-8",
    )

    result = inspect_project(project)

    assert result["text_quality"]["finding_count"] == 1
    finding = result["text_quality"]["findings"][0]
    assert finding["file"] == "README.md"
    assert finding["category"] == "production_validation_overclaim"
    assert result["status"]["overall"] == "WARN"



def test_context_rebuild_script_is_excluded_from_quality_findings(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project / "scripts/rebuild_day13_api_core_context.py",
        "\n".join(
            [
                'MOJIBAKE_TOKENS = (',
                '    "媛",',
                '    "吏",',
                '    "쒖",',
                '    "湲",',
                '    "怨",',
                ')',
            ]
        )
        + "\n",
    )

    result = inspect_project(project)

    assert result["schema_version"] == 5
    assert result["text_quality"]["finding_count"] == 0
    assert result["status"]["warning_count"] == 0


def test_diagnostic_backup_txt_is_excluded_from_active_diagnostic_info(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project
        / "reports/artifacts/backups/"
        "day13_api_core_context.before_utf8_rebuild.deadbeef.txt",
        "吏媛 쒖湲怨\n",
    )

    result = inspect_project(project)

    diagnostic = result["text_quality"]["diagnostic_artifacts"]
    assert diagnostic["file_count"] == 0
    assert diagnostic["excluded_backup_count"] == 1
    assert diagnostic["excluded_backup_files"] == [
        (
            "reports/artifacts/backups/"
            "day13_api_core_context.before_utf8_rebuild.deadbeef.txt"
        )
    ]
    assert result["status"]["information_count"] == 0



def test_day14_docs_generator_forbidden_phrase_constants_are_excluded(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project / "scripts/create_day14_docs.py",
        "\n".join(
            [
                "FORBIDDEN_OVERCLAIMS = (",
                '    "실제 생산 환경에서 검증 완료",',
                '    "산업 현장 배포 완료",',
                '    "실시간 생산 시스템 구축",',
                '    "COCO 공식 mAP와 완전히 동일",',
                '    "OpenCV Contour가 실제 결함 위치",',
                '    "Detection Prediction이 Ground Truth",',
                ")",
            ]
        )
        + "\n",
    )

    result = inspect_project(project)

    assert result["schema_version"] == 5
    assert result["text_quality"]["finding_count"] == 0
    assert result["status"]["warning_count"] == 0


def test_actual_readme_overclaim_is_still_detected_with_v5(
    tmp_path: Path,
) -> None:
    project = _build_minimal_project(tmp_path)
    _write(
        project / "README.md",
        "# Project\n\n실제 생산 환경에서 검증 완료\n",
    )

    result = inspect_project(project)

    assert result["schema_version"] == 5
    assert result["text_quality"]["finding_count"] == 1
    assert result["status"]["warning_count"] == 1

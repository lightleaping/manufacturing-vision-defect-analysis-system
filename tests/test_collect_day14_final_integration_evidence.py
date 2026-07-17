"""Tests for Day 14 final-integration evidence collection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_day14_final_integration_evidence import (
    EvidenceCollectionError,
    build_architecture_mermaid,
    build_markdown,
    build_user_flow_mermaid,
    collect_evidence,
    extract_metrics_from_json,
    inspect_fastapi_endpoints,
    inspect_readme,
    select_artifacts,
    write_outputs,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"

    _write(
        root / "README.md",
        "\n".join(
            [
                "# Manufacturing Vision Defect Analysis System",
                "",
                "## Overview",
                "",
                "<!-- DAY13_SECTION_START -->",
                "Day 13",
                "<!-- DAY13_SECTION_END -->",
            ]
        )
        + "\n",
    )
    _write(root / "requirements.txt", "pytest\n")
    _write(root / ".gitignore", ".venv/\n")

    _write(
        root / "src/api/app.py",
        """
from fastapi import FastAPI
app = FastAPI()
API_PREFIX = "/api/v1"

@app.get(API_PREFIX + "/health")
def health():
    return {}

@app.post(f"{API_PREFIX}/predictions")
def predictions():
    return {}

@app.post("/api/v1/detection/predictions")
def detection_predictions():
    return {}
""".strip()
        + "\n",
    )
    _write(root / "src/dashboard/app.py", "import streamlit as st\n")
    _write(
        root / "src/dashboard/pages/2_Detection.py",
        "from src.dashboard.detection_page import render\n",
    )
    _write(
        root / "src/dashboard/detection_page.py",
        "def render():\n    return None\n",
    )
    _write(
        root / "src/dashboard/detection_api_client.py",
        "class Client:\n    pass\n",
    )
    _write(root / "tests/test_example.py", "def test_ok():\n    assert True\n")
    _write(root / "scripts/example.py", "VALUE = 1\n")

    _write(
        root / "models/checkpoints/resnet18_transfer_best.pt",
        "checkpoint",
    )
    _write(
        root / "models/detection/day12_detection_best.pt",
        "checkpoint",
    )
    _write(
        root / "models/detection/day12_detection_latest.pt",
        "checkpoint",
    )
    _write(
        root / "data/processed/neu_det/splits.json",
        "{}\n",
    )

    artifact_root = root / "reports/artifacts"
    _write(
        artifact_root / "day4_resnet18_test_evaluation.json",
        json.dumps(
            {
                "metrics": {
                    "accuracy": 0.9734,
                    "precision": 0.9717,
                    "recall": 0.9868,
                    "f1_score": 0.9792,
                },
                "confusion_matrix": {
                    "tn": 249,
                    "fp": 13,
                    "fn": 6,
                    "tp": 447,
                },
            }
        ),
    )
    _write(
        artifact_root / "day12_detection_test_evaluation.json",
        json.dumps(
            {
                "precision": 0.81295,
                "recall": 0.526807,
                "f1": 0.639321,
                "map_50": 0.707726,
            }
        ),
    )
    _write(
        artifact_root / "day12_detection_failure_analysis.json",
        json.dumps(
            {
                "images_with_failures": 129,
                "failure_events": 229,
            }
        ),
    )
    _write(
        artifact_root / "day13_detection_api_smoke_test.json",
        json.dumps({"status": "PASS"}),
    )
    _write(
        artifact_root / "day13_detection_dashboard_api_client_validation.json",
        json.dumps({"status": "PASS"}),
    )
    _write(
        artifact_root / "day13_detection_integration_summary.json",
        json.dumps(
            {
                "tests": {
                    "regression_test_count": 1668,
                    "warning_count": 1,
                },
                "manual_browser_check_status": "not_recorded",
            }
        ),
    )
    _write(
        artifact_root / "day14_final_integration_prerequisites_inspection.json",
        json.dumps({"status": {"overall": "PASS"}}),
    )

    _write(root / "reports/day13_summary.md", "# Day 13\n")
    _write(root / "reports/figures/example.png", "png")

    return root


def test_readme_inspection_collects_headings_and_markers(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)

    result = inspect_readme(project)

    assert result["heading_count"] == 2
    assert result["marker_days"] == [13]
    assert result["duplicate_markers"] == []


def test_endpoint_inspection_supports_constants_and_fstrings(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)

    result = inspect_fastapi_endpoints(project)

    assert result["missing_expected"] == []
    assert result["expected_status"] == {
        "GET /api/v1/health": True,
        "POST /api/v1/predictions": True,
        "POST /api/v1/detection/predictions": True,
    }


def test_artifact_selection_uses_filename_hints(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)

    result = select_artifacts(project)

    assert result["classification_evaluation"] == [
        "reports/artifacts/day4_resnet18_test_evaluation.json"
    ]
    assert result["detection_failure_analysis"] == [
        "reports/artifacts/day12_detection_failure_analysis.json"
    ]


def test_metric_extraction_finds_nested_values(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    path = (
        project
        / "reports/artifacts/day4_resnet18_test_evaluation.json"
    )

    result = extract_metrics_from_json(path)

    assert result["accuracy"][0]["value"] == 0.9734
    assert result["f1"][0]["value"] == 0.9792
    assert result["tn"][0]["value"] == 249
    assert result["tp"][0]["value"] == 447


def test_collect_evidence_has_read_only_policy(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    readme = project / "README.md"
    before = readme.read_bytes()

    result = collect_evidence(project)

    assert readme.read_bytes() == before
    assert result["policy"] == {
        "readme_modified": False,
        "source_modified": False,
        "training_executed": False,
        "inference_executed": False,
        "new_dependency_added": False,
    }


def test_collect_evidence_passes_minimal_complete_project(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)

    result = collect_evidence(project)

    assert result["status"]["overall"] == "PASS"
    assert result["status"]["error_count"] == 0
    assert result["status"]["warning_count"] == 0


def test_collect_evidence_fails_on_duplicate_marker(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    readme = project / "README.md"
    _write(
        readme,
        readme.read_text(encoding="utf-8")
        + "<!-- DAY13_SECTION_START -->\n",
    )

    result = collect_evidence(project)

    assert result["status"]["overall"] == "FAIL"
    assert result["status"]["error_count"] == 1


def test_collect_evidence_warns_when_important_path_missing(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    (
        project
        / "models/detection/day12_detection_best.pt"
    ).unlink()

    result = collect_evidence(project)

    assert result["status"]["overall"] == "WARN"
    assert any(
        "detection_checkpoint_best" in warning
        for warning in result["status"]["warnings"]
    )


def test_architecture_diagram_separates_three_pipelines() -> None:
    diagram = build_architecture_mermaid()

    assert "Classification Pipeline" in diagram
    assert "Object Detection Pipeline" in diagram
    assert "OpenCV Auxiliary Analysis" in diagram
    assert "OpenCV Contours are not Ground Truth" in diagram


def test_user_flow_documents_api_boundary() -> None:
    diagram = build_user_flow_mermaid()

    assert "Streamlit does not load checkpoints directly" in diagram
    assert "Model is loaded once in FastAPI lifespan" in diagram


def test_markdown_contains_required_readme_sections(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    evidence = collect_evidence(project)

    markdown = build_markdown(evidence)

    assert "README Final Structure Proposal" in markdown
    assert "System Architecture" in markdown
    assert "End-to-End User Flow" in markdown
    assert "Classification·OpenCV·Detection Role Boundary" in markdown
    assert "README Modification Gate" in markdown


def test_write_outputs_creates_utf8_json_and_markdown(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    evidence = collect_evidence(project)

    json_path, markdown_path = write_outputs(
        project_root=project,
        evidence=evidence,
        json_output=Path(
            "reports/artifacts/day14_final_integration_evidence.json"
        ),
        markdown_output=Path(
            "reports/day14_final_integration_readme_architecture_plan.md"
        ),
    )

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert saved["status"]["overall"] == "PASS"
    assert "# Day 14" in markdown


def test_invalid_selected_json_is_reported_as_warning(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    _write(
        project
        / "reports/artifacts/day12_detection_failure_bad.json",
        "{invalid",
    )

    result = collect_evidence(project)

    assert result["status"]["overall"] == "WARN"
    assert any(
        "day12_detection_failure_bad.json" in warning
        for warning in result["status"]["warnings"]
    )


def test_missing_readme_raises_clear_error(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    (project / "README.md").unlink()

    with pytest.raises(EvidenceCollectionError, match="README.md"):
        collect_evidence(project)



def test_actual_classification_checkpoint_path_is_detected(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)

    result = collect_evidence(project)

    checkpoint = result["important_paths"]["classification_checkpoint"]
    assert checkpoint["found_any"] is True
    assert checkpoint["found"] == [
        "models/checkpoints/resnet18_transfer_best.pt"
    ]


def test_missing_classification_checkpoint_is_reported_as_warning(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    (
        project
        / "models/checkpoints/resnet18_transfer_best.pt"
    ).unlink()

    result = collect_evidence(project)

    assert result["status"]["overall"] == "WARN"
    assert any(
        "classification_checkpoint" in warning
        for warning in result["status"]["warnings"]
    )

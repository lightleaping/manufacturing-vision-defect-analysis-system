from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "inspect_day13_integration_prerequisites.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("day13_inspector", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_role_hints_detect_fastapi_router_schema_and_streamlit(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "api_page.py"
    text = """
from fastapi import APIRouter
from pydantic import BaseModel
import streamlit as st

router = APIRouter()

class Response(BaseModel):
    value: int

@router.post("/api/v1/detection/predictions")
def predict():
    st.file_uploader("image")
"""
    hints = module.role_hints_for(path, text)

    assert "router" in hints
    assert "schema" in hints
    assert "streamlit_page" in hints
    assert "detection" in hints


def test_inspect_file_extracts_endpoint_and_pydantic_model(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "router.py"
    source.write_text(
        """
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class DetectionResponse(BaseModel):
    detection_count: int

@router.post("/api/v1/detection/predictions")
def predict():
    return {"detection_count": 0}
""",
        encoding="utf-8",
    )

    record = module.inspect_file(source, tmp_path)

    assert record.pydantic_models == ["DetectionResponse"]
    assert record.routes == [
        {
            "object": "router",
            "method": "POST",
            "path": "/api/v1/detection/predictions",
        }
    ]


def test_read_readme_status_counts_day12_markers(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "<!-- DAY12_DETECTION_TRAINING_EVALUATION_START -->",
                "content",
                "<!-- DAY12_DETECTION_TRAINING_EVALUATION_END -->",
            ]
        ),
        encoding="utf-8",
    )

    result = module.read_readme_status(tmp_path)

    assert result["day12_start_marker_count"] == 1
    assert result["day12_end_marker_count"] == 1


def test_pytest_cache_status_reads_cached_nodeids(tmp_path: Path) -> None:
    module = load_module()
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "nodeids").write_text(
        json.dumps(["tests/test_a.py::test_a", "tests/test_b.py::test_b"]),
        encoding="utf-8",
    )
    (cache / "lastfailed").write_text("{}", encoding="utf-8")

    result = module.pytest_cache_status(tmp_path)

    assert result["cached_collected_test_count"] == 2
    assert result["cached_lastfailed_count"] == 0


def test_validate_expected_items_reports_missing_items(tmp_path: Path) -> None:
    module = load_module()
    summary = {
        "detected_endpoints": [],
        "files_by_role": {},
        "pydantic_models": [],
        "include_router_calls": [],
        "exception_handlers": [],
        "streamlit_files": [],
    }

    result = module.validate_expected_items(
        project_root=tmp_path,
        checkpoint={"exists": False},
        readme={},
        summary=summary,
    )

    assert result["all_passed"] is False
    assert result["checks"]["checkpoint_exists"] is False
    assert result["checks"]["classification_prediction_endpoint_detected"] is False

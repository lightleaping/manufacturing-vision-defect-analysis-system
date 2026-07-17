"""Day 13 Detection Dashboard Stage 2 정적 구조 점검."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    Path("src/dashboard/detection_api_client.py"),
    Path("src/dashboard/detection_session_state.py"),
    Path("src/dashboard/detection_ui_helpers.py"),
    Path("src/dashboard/detection_page.py"),
    Path("src/dashboard/pages/2_Detection.py"),
    Path("tests/test_detection_dashboard_api_client.py"),
    Path("tests/test_detection_dashboard_session_state.py"),
    Path("tests/test_detection_dashboard_ui_helpers.py"),
    Path("tests/test_detection_dashboard_page.py"),
)

DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day13_detection_dashboard_stage2_inspection.json"
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8")
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
    return modules


def run_inspection(
    *,
    project_root: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    page_path = (
        root
        / "src"
        / "dashboard"
        / "detection_page.py"
    )
    client_path = (
        root
        / "src"
        / "dashboard"
        / "detection_api_client.py"
    )

    file_checks = {
        path.as_posix(): (root / path).is_file()
        for path in REQUIRED_FILES
    }
    page_source = (
        page_path.read_text(encoding="utf-8")
        if page_path.is_file()
        else ""
    )
    client_source = (
        client_path.read_text(encoding="utf-8")
        if client_path.is_file()
        else ""
    )
    page_imports = (
        imported_modules(page_path)
        if page_path.is_file()
        else set()
    )

    forbidden_imports = sorted(
        module
        for module in page_imports
        if module.startswith(
            (
                "torch",
                "torchvision",
                "src.api.detection",
                "src.detection",
            )
        )
    )

    checks = {
        "required_files_exist": all(
            file_checks.values()
        ),
        "detection_endpoint_present": (
            "/api/v1/detection/predictions"
            in client_source
        ),
        "default_threshold_is_0_5": (
            "value=0.50"
            in page_source
        ),
        "threshold_range_present": (
            "min_value=0.05"
            in page_source
            and "max_value=0.95"
            in page_source
        ),
        "api_client_only": not forbidden_imports,
        "prediction_overlay_present": (
            "render_detection_overlay"
            in page_source
        ),
        "prediction_table_present": (
            "Prediction Table"
            in page_source
        ),
        "opencv_distinction_present": (
            "OpenCV"
            in page_source
            and "Contour 후보"
            in page_source
        ),
        "ground_truth_warning_present": (
            "Ground Truth"
            in page_source
        ),
    }

    result: dict[str, Any] = {
        "stage": "day13_detection_dashboard_stage2",
        "project_root": str(root),
        "files": file_checks,
        "page_imports": sorted(page_imports),
        "forbidden_imports": forbidden_imports,
        "checks": checks,
        "validation_passed": all(checks.values()),
    }

    output = (
        artifact_path
        if artifact_path.is_absolute()
        else root / artifact_path
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DAY 13 - DETECTION DASHBOARD STAGE 2 INSPECTION")
    print("=" * 100)
    print(f"Project root             : {root}")
    for name, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{name}"
        )
    print(f"Artifact                 : {output}")
    print(
        "[RESULT]                 : "
        + (
            "PASS"
            if result["validation_passed"]
            else "FAIL"
        )
    )
    print("=" * 100)

    if not result["validation_passed"]:
        raise RuntimeError(
            "Day 13 Detection Dashboard Stage 2 inspection failed."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT,
    )
    args = parser.parse_args()

    run_inspection(
        project_root=args.project_root,
        artifact_path=args.artifact_path,
    )


if __name__ == "__main__":
    main()

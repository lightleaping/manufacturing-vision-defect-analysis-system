"""Collect Day 14 final-integration evidence and generate a README/architecture plan.

This script is intentionally read-only for the existing application and README.
It creates two new outputs:

- reports/artifacts/day14_final_integration_evidence.json
- reports/day14_final_integration_readme_architecture_plan.md

The outputs are based on the current repository structure and existing JSON
artifacts. No model training or inference is performed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KO = "제조 비전 결함 분석 시스템"

DEFAULT_JSON_OUTPUT = Path(
    "reports/artifacts/day14_final_integration_evidence.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "reports/day14_final_integration_readme_architecture_plan.md"
)

README_HEADING_PATTERN = re.compile(
    r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)
DAY_MARKER_PATTERN = re.compile(
    r"<!--\s*(DAY(?P<day>\d+)_[A-Z0-9_]+)_(?P<kind>START|END)\s*-->",
    re.IGNORECASE,
)

EXPECTED_ENDPOINTS = (
    ("GET", "/api/v1/health"),
    ("POST", "/api/v1/predictions"),
    ("POST", "/api/v1/detection/predictions"),
)

IMPORTANT_PATHS = {
    "readme": ("README.md",),
    "requirements": ("requirements.txt",),
    "gitignore": (".gitignore",),
    "fastapi_app": ("src/api/app.py",),
    "classification_dashboard": ("src/dashboard/app.py",),
    "detection_dashboard": (
        "src/dashboard/pages/2_Detection.py",
        "src/dashboard/detection_page.py",
    ),
    "detection_api_client": (
        "src/dashboard/detection_api_client.py",
    ),
    "classification_checkpoint": (
        "models/checkpoints/resnet18_transfer_best.pt",
        "models/day4_resnet18_best.pt",
        "models/classification/day4_resnet18_best.pt",
        "models/resnet18_best.pt",
    ),
    "detection_checkpoint_best": (
        "models/detection/day12_detection_best.pt",
    ),
    "detection_checkpoint_latest": (
        "models/detection/day12_detection_latest.pt",
    ),
    "detection_split_manifest": (
        "data/processed/neu_det/splits.json",
    ),
    "day13_summary": (
        "reports/artifacts/day13_detection_integration_summary.json",
    ),
    "day14_prerequisites": (
        "reports/artifacts/day14_final_integration_prerequisites_inspection.json",
    ),
}

ARTIFACT_HINTS = {
    "classification_evaluation": (
        "day4",
        "test",
        "evaluation",
    ),
    "detection_evaluation": (
        "day12",
        "detection",
        "evaluation",
    ),
    "detection_failure_analysis": (
        "day12",
        "failure",
    ),
    "day13_api_smoke": (
        "day13",
        "api",
        "smoke",
    ),
    "day13_dashboard_validation": (
        "day13",
        "dashboard",
        "validation",
    ),
    "day13_integration_summary": (
        "day13",
        "integration",
        "summary",
    ),
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "test_accuracy"),
    "precision": ("precision", "test_precision"),
    "recall": ("recall", "test_recall"),
    "f1": ("f1", "f1_score", "test_f1"),
    "tn": ("tn", "true_negative", "true_negatives"),
    "fp": ("fp", "false_positive", "false_positives"),
    "fn": ("fn", "false_negative", "false_negatives"),
    "tp": ("tp", "true_positive", "true_positives"),
    "map_50": (
        "map_50",
        "map50",
        "map@0.50",
        "mAP@0.50",
        "test_map_50",
    ),
    "project_map_50_95": (
        "project_map_50_95",
        "project_map@0.50:0.95",
        "map_50_95",
    ),
    "mean_matched_iou": (
        "mean_matched_iou",
        "matched_iou_mean",
    ),
    "test_images": (
        "test_images",
        "image_count",
        "total_images",
    ),
    "ground_truth_boxes": (
        "ground_truth_boxes",
        "gt_boxes",
        "total_ground_truth_boxes",
    ),
    "images_with_failures": (
        "images_with_failures",
        "failure_image_count",
    ),
    "failure_events": (
        "failure_events",
        "failure_event_count",
        "total_failure_events",
    ),
    "checkpoint_epoch": (
        "checkpoint_epoch",
        "best_epoch",
        "epoch",
    ),
    "checkpoint_metric_name": (
        "checkpoint_metric_name",
        "best_metric_name",
    ),
    "checkpoint_metric_value": (
        "checkpoint_metric_value",
        "best_metric_value",
    ),
    "targeted_test_count": (
        "targeted_test_count",
        "target_test_count",
    ),
    "regression_test_count": (
        "regression_test_count",
        "test_count",
    ),
    "warning_count": ("warning_count",),
    "runtime_seconds": (
        "runtime_seconds",
        "test_runtime_seconds",
    ),
    "manual_browser_check_status": (
        "manual_browser_check_status",
    ),
}

NEGATION_CONTEXT_PATTERN = re.compile(
    r"(아님|아니다|않|금지|동일하지|not|never|isn't|is not)",
    re.IGNORECASE,
)


class EvidenceCollectionError(RuntimeError):
    """Evidence collection cannot continue safely."""


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvidenceCollectionError(
            f"UTF-8로 읽을 수 없는 파일입니다: {path}"
        ) from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise EvidenceCollectionError(
            f"JSON 형식이 잘못됐습니다: {path} "
            f"(line={exc.lineno}, column={exc.colno})"
        ) from exc


def _write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            text,
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_readme(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    if not readme.is_file():
        raise EvidenceCollectionError("README.md가 없습니다.")

    text = _read_text(readme)
    headings = [
        {
            "level": len(match.group("level")),
            "title": match.group("title").strip(),
            "line": text[: match.start()].count("\n") + 1,
        }
        for match in README_HEADING_PATTERN.finditer(text)
    ]

    marker_rows: list[dict[str, Any]] = []
    for match in DAY_MARKER_PATTERN.finditer(text):
        marker_rows.append(
            {
                "day": int(match.group("day")),
                "name": match.group(1).upper(),
                "kind": match.group("kind").upper(),
                "line": text[: match.start()].count("\n") + 1,
            }
        )

    marker_counter = Counter(
        (row["name"], row["kind"])
        for row in marker_rows
    )

    return {
        "path": "README.md",
        "size_bytes": readme.stat().st_size,
        "sha256": _sha256(readme),
        "line_count": len(text.splitlines()),
        "headings": headings,
        "heading_count": len(headings),
        "day_markers": marker_rows,
        "marker_count": len(marker_rows),
        "marker_days": sorted({row["day"] for row in marker_rows}),
        "duplicate_markers": [
            {
                "name": name,
                "kind": kind,
                "count": count,
            }
            for (name, kind), count in sorted(marker_counter.items())
            if count != 1
        ],
    }


def inspect_important_paths(root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for name, candidates in IMPORTANT_PATHS.items():
        existing = [
            candidate
            for candidate in candidates
            if (root / candidate).is_file()
        ]
        results[name] = {
            "found": existing,
            "found_any": bool(existing),
            "candidates": list(candidates),
        }

    return results


def _safe_static_string(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _safe_static_string(node.left, constants)
        right = _safe_static_string(node.right, constants)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                resolved = _safe_static_string(value.value, constants)
                if resolved is None:
                    return None
                parts.append(resolved)
            else:
                return None
        return "".join(parts)
    return None


def inspect_fastapi_endpoints(root: Path) -> dict[str, Any]:
    endpoints: list[dict[str, Any]] = []

    for path in sorted((root / "src").rglob("*.py")):
        text = _read_text(path)
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise EvidenceCollectionError(
                f"Python 문법 오류: {path}:{exc.lineno} {exc.msg}"
            ) from exc

        constants: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and isinstance(getattr(node, "value", None), ast.AST)
            ):
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                value = _safe_static_string(node.value, constants)
                if value is None:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if not isinstance(decorator.func, ast.Attribute):
                    continue
                method = decorator.func.attr.upper()
                if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    continue
                if not decorator.args:
                    continue
                endpoint_path = _safe_static_string(
                    decorator.args[0],
                    constants,
                )
                if endpoint_path is None:
                    continue
                endpoints.append(
                    {
                        "method": method,
                        "path": endpoint_path,
                        "file": _relative(path, root),
                        "line": node.lineno,
                        "function": node.name,
                    }
                )

    endpoints.sort(
        key=lambda row: (
            row["path"],
            row["method"],
            row["file"],
            row["line"],
        )
    )
    pairs = {(row["method"], row["path"]) for row in endpoints}

    return {
        "endpoints": endpoints,
        "expected_status": {
            f"{method} {path}": (method, path) in pairs
            for method, path in EXPECTED_ENDPOINTS
        },
        "missing_expected": [
            f"{method} {path}"
            for method, path in EXPECTED_ENDPOINTS
            if (method, path) not in pairs
        ],
    }


def list_inventory(root: Path) -> dict[str, Any]:
    def files(directory: str, suffix: str | None = None) -> list[str]:
        target = root / directory
        if not target.exists():
            return []
        rows = [
            _relative(path, root)
            for path in sorted(target.rglob("*"))
            if path.is_file()
            and (suffix is None or path.suffix.lower() == suffix)
        ]
        return rows

    reports = files("reports", ".md")
    artifacts = files("reports/artifacts")
    figures = files("reports/figures")
    source_files = files("src")
    tests = files("tests", ".py")
    scripts = files("scripts", ".py")

    return {
        "counts": {
            "reports": len(reports),
            "artifacts": len(artifacts),
            "figures": len(figures),
            "source_files": len(source_files),
            "test_files": len(tests),
            "script_files": len(scripts),
        },
        "reports": reports,
        "artifacts": artifacts,
        "figures": figures,
        "source_files": source_files,
        "tests": tests,
        "scripts": scripts,
    }


def _filename_matches(path: Path, tokens: Sequence[str]) -> bool:
    lowered = path.stem.casefold()
    return all(token.casefold() in lowered for token in tokens)


def select_artifacts(root: Path) -> dict[str, list[str]]:
    artifact_root = root / "reports" / "artifacts"
    json_files = (
        sorted(artifact_root.rglob("*.json"))
        if artifact_root.exists()
        else []
    )

    selected: dict[str, list[str]] = {}
    for category, tokens in ARTIFACT_HINTS.items():
        selected[category] = [
            _relative(path, root)
            for path in json_files
            if _filename_matches(path, tokens)
        ]

    return selected


def _walk_json(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, str(key), child
            yield from _walk_json(child, prefix=path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            yield from _walk_json(child, prefix=path)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def extract_metrics_from_json(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(path)
    normalized_aliases = {
        metric: {_normalize_key(alias) for alias in aliases}
        for metric, aliases in METRIC_ALIASES.items()
    }

    found: dict[str, list[dict[str, Any]]] = {
        metric: []
        for metric in METRIC_ALIASES
    }

    for json_path, key, value in _walk_json(payload):
        normalized_key = _normalize_key(key)
        for metric, aliases in normalized_aliases.items():
            if normalized_key in aliases:
                found[metric].append(
                    {
                        "json_path": json_path,
                        "value": value,
                    }
                )

    return {
        metric: rows
        for metric, rows in found.items()
        if rows
    }


def collect_artifact_metrics(
    root: Path,
    selected: dict[str, list[str]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}

    for category, paths in selected.items():
        category_rows: list[dict[str, Any]] = []
        for relative_path in paths:
            path = root / relative_path
            try:
                metrics = extract_metrics_from_json(path)
                category_rows.append(
                    {
                        "path": relative_path,
                        "valid_json": True,
                        "metrics": metrics,
                    }
                )
            except EvidenceCollectionError as exc:
                category_rows.append(
                    {
                        "path": relative_path,
                        "valid_json": False,
                        "error": str(exc),
                        "metrics": {},
                    }
                )
        rows[category] = category_rows

    return rows


def build_run_commands(paths: dict[str, Any]) -> list[dict[str, Any]]:
    commands = [
        {
            "name": "전체 테스트",
            "command": (
                ".\\.venv\\Scripts\\python.exe `\n"
                "    -m pytest `\n"
                "    -q"
            ),
            "static_prerequisite": paths["readme"]["found_any"],
        },
        {
            "name": "FastAPI",
            "command": (
                ".\\.venv\\Scripts\\python.exe `\n"
                "    -m uvicorn `\n"
                "    src.api.app:app `\n"
                "    --host 127.0.0.1 `\n"
                "    --port 8000"
            ),
            "static_prerequisite": paths["fastapi_app"]["found_any"],
        },
        {
            "name": "Classification Dashboard",
            "command": (
                ".\\.venv\\Scripts\\python.exe `\n"
                "    -m streamlit `\n"
                "    run `\n"
                "    .\\src\\dashboard\\app.py"
            ),
            "static_prerequisite": paths[
                "classification_dashboard"
            ]["found_any"],
        },
        {
            "name": "Detection Dashboard",
            "command": (
                ".\\.venv\\Scripts\\python.exe `\n"
                "    -m streamlit `\n"
                "    run `\n"
                "    .\\src\\dashboard\\app.py"
            ),
            "static_prerequisite": (
                paths["classification_dashboard"]["found_any"]
                and paths["detection_dashboard"]["found_any"]
            ),
            "note": (
                "Streamlit Multipage 구조이므로 app.py 실행 후 "
                "Detection 페이지를 선택한다."
            ),
        },
    ]
    return commands


def build_architecture_mermaid() -> str:
    return """```mermaid
flowchart LR
    U[Browser / User]

    subgraph Classification["Classification Pipeline"]
        CD[Streamlit Classification]
        CA[POST /api/v1/predictions]
        CM[ResNet18 Transfer]
        CO[NORMAL / DEFECT]
        CD --> CA --> CM --> CO
    end

    subgraph Detection["Object Detection Pipeline"]
        DD[Streamlit Detection]
        DA[POST /api/v1/detection/predictions]
        DM[Faster R-CNN MobileNetV3 320 FPN]
        DO[Class / Score / Bounding Box]
        DD --> DA --> DM --> DO
    end

    subgraph OpenCV["OpenCV Auxiliary Analysis"]
        OI[Input Image]
        OP[Brightness / Histogram / Edge]
        OM[Threshold / Morphology]
        OC[Contour Candidates]
        OI --> OP --> OM --> OC
    end

    U --> CD
    U --> DD

    CData[Casting Product Image Dataset] --> CTrain[Classification Training]
    CTrain --> CCheckpoint[ResNet18 Checkpoint]
    CCheckpoint --> CM
    CCheckpoint --> CEval[Classification Evaluation]

    DData[NEU-DET Dataset] --> DTrain[Detection Training]
    DTrain --> DCheckpoint[Detection Best Checkpoint]
    DCheckpoint --> DM
    DCheckpoint --> DEval[Detection Evaluation]
    DEval --> DFail[Failure Analysis]

    CEval --> Artifacts[Reports / JSON Artifacts / Figures]
    DFail --> Artifacts

    Note1["OpenCV Contours are not Ground Truth or Detection Predictions"]
    OC -. meaning .-> Note1
```"""


def build_user_flow_mermaid() -> str:
    return """```mermaid
sequenceDiagram
    actor User
    participant ST as Streamlit
    participant API as FastAPI
    participant Model as Loaded Model Service

    User->>ST: Upload image and request analysis
    ST->>API: Multipart HTTP request
    API->>API: Validate extension, MIME, decode, size, pixels
    API->>Model: Inference with process-local loaded checkpoint
    Model-->>API: Prediction result
    API-->>ST: Validated response schema
    ST-->>User: Label or Detection overlay/table

    Note over ST,Model: Streamlit does not load checkpoints directly
    Note over API,Model: Model is loaded once in FastAPI lifespan
```"""


def _markdown_list(items: Sequence[str]) -> str:
    if not items:
        return "- 확인된 항목 없음"
    return "\n".join(f"- `{item}`" for item in items)


def build_markdown(
    evidence: dict[str, Any],
) -> str:
    readme = evidence["readme"]
    inventory = evidence["inventory"]
    selected = evidence["selected_artifacts"]
    endpoints = evidence["fastapi"]["endpoints"]
    commands = evidence["run_commands"]

    heading_rows = "\n".join(
        f"- L{row['line']}: {'#' * row['level']} {row['title']}"
        for row in readme["headings"]
    ) or "- Heading 없음"

    endpoint_rows = "\n".join(
        f"- `{row['method']} {row['path']}` — "
        f"`{row['file']}:{row['line']}`"
        for row in endpoints
    ) or "- Endpoint 없음"

    command_sections = []
    for row in commands:
        note = f"\n\n{row['note']}" if row.get("note") else ""
        command_sections.append(
            f"### {row['name']}\n\n"
            f"정적 경로 확인: "
            f"`{'PASS' if row['static_prerequisite'] else 'NOT FOUND'}`\n\n"
            f"```powershell\n{row['command']}\n```"
            f"{note}"
        )

    selected_sections = []
    for category, paths in selected.items():
        selected_sections.append(
            f"### {category}\n\n{_markdown_list(paths)}"
        )

    return f"""# Day 14 — Final Integration README and Architecture Plan

> 이 문서는 현재 저장소와 기존 Artifact를 읽어 생성한 **README 수정 전 근거 문서**다.
> README 자체와 Application Source는 수정하지 않았다.

## 1. Evidence Summary

- 프로젝트: **{PROJECT_NAME}**
- 한글명: **{PROJECT_NAME_KO}**
- README Heading 수: **{readme['heading_count']}**
- README Marker 수: **{readme['marker_count']}**
- Report 수: **{inventory['counts']['reports']}**
- Artifact 수: **{inventory['counts']['artifacts']}**
- Figure 수: **{inventory['counts']['figures']}**
- Source 파일 수: **{inventory['counts']['source_files']}**
- Test 파일 수: **{inventory['counts']['test_files']}**
- Script 파일 수: **{inventory['counts']['script_files']}**

## 2. Existing README Headings

{heading_rows}

## 3. README Final Structure Proposal

1. Project Overview
2. Problem Definition
3. Why This Project
4. Core Features
5. System Architecture
6. End-to-End User Flow
7. Project Structure
8. Environment
9. Dataset
10. Classification Pipeline
11. OpenCV Analysis Pipeline
12. Object Detection Pipeline
13. FastAPI Endpoints
14. Streamlit Dashboard
15. Model Training Policy
16. Evaluation Results
17. Failure Analysis
18. Validation and Testing
19. How to Run
20. Key Design Decisions
21. Limitations
22. Future Improvements
23. Portfolio Summary

### 정리 원칙

- 기존 Day 1~13 Marker는 제거하거나 중복 생성하지 않는다.
- Classification과 Detection의 Dataset·Label·Model·지표를 분리한다.
- OpenCV Contour는 Ground Truth 또는 Detection Prediction으로 표현하지 않는다.
- Detection의 `Project mAP@0.50:0.95`는 프로젝트 내부 all-point AP로 설명한다.
- 수동 Browser 검증은 실제 기록이 없으면 `not_recorded`로 유지한다.
- 최종 테스트 수는 마지막 전체 회귀 테스트 출력 후 반영한다.

## 4. System Architecture

{build_architecture_mermaid()}

## 5. End-to-End User Flow

{build_user_flow_mermaid()}

## 6. Verified FastAPI Endpoints

{endpoint_rows}

## 7. Classification·OpenCV·Detection Role Boundary

### Classification

- 입력 이미지 전체를 `NORMAL / DEFECT`로 분류한다.
- 결함 위치나 세부 결함 Class는 반환하지 않는다.
- Casting Product Image Dataset과 ResNet18 Checkpoint를 사용한다.

### OpenCV

- 이미지의 명암·히스토그램·경계·Threshold·Morphology 특성을 분석한다.
- Contour는 Threshold·Morphology 기반 후보 영역이다.
- Contour는 Ground Truth도 Detection Prediction도 아니다.

### Detection

- NEU-DET 이미지에서 결함 Class·Score·Bounding Box를 예측한다.
- Faster R-CNN MobileNetV3 Large 320 FPN Checkpoint를 사용한다.
- Detection Prediction은 Ground Truth가 아니다.
- Detection 0개는 현재 Threshold 이상 Prediction이 없다는 의미다.

## 8. Selected Evidence Artifacts

{chr(10).join(selected_sections)}

## 9. Static Run Command Plan

{chr(10).join(command_sections)}

## 10. README Modification Gate

다음 항목이 모두 충족된 뒤 README를 수정한다.

- [ ] Classification 최종 평가 Artifact의 실제 키와 수치 확정
- [ ] Detection 평가·Failure Analysis Artifact의 실제 키와 수치 확정
- [ ] Day 13 API·Dashboard 검증 상태 확정
- [ ] 실행 Command 정적 경로 확인
- [ ] Architecture와 User Flow 표현 검토
- [ ] Portfolio·이력서·면접 문구 작성
- [ ] Day 14 대상 테스트 실행
- [ ] 전체 회귀 테스트 실행
- [ ] 마지막 전체 회귀 테스트 수로 README 최종 갱신

## 11. Current Safety Status

- README 수정: **아니오**
- 기존 Source 수정: **아니오**
- 모델 학습·추론 실행: **아니오**
- 기존 Checkpoint 변경: **아니오**
- 신규 Dependency: **없음**
"""


def derive_status(evidence: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if evidence["readme"]["duplicate_markers"]:
        errors.append("README Marker 중복 또는 개수 불일치가 있습니다.")

    missing_endpoints = evidence["fastapi"]["missing_expected"]
    if missing_endpoints:
        errors.append(
            "예상 FastAPI Endpoint를 찾지 못했습니다: "
            + ", ".join(missing_endpoints)
        )

    for name in (
        "readme",
        "fastapi_app",
        "classification_dashboard",
        "detection_dashboard",
        "detection_api_client",
        "classification_checkpoint",
        "detection_checkpoint_best",
        "detection_split_manifest",
        "day13_summary",
        "day14_prerequisites",
    ):
        if not evidence["important_paths"][name]["found_any"]:
            warnings.append(f"주요 경로를 찾지 못했습니다: {name}")

    invalid_artifacts = [
        row["path"]
        for rows in evidence["artifact_metrics"].values()
        for row in rows
        if not row["valid_json"]
    ]
    if invalid_artifacts:
        warnings.append(
            "읽을 수 없는 선택 Artifact가 있습니다: "
            + ", ".join(invalid_artifacts)
        )

    if errors:
        overall = "FAIL"
    elif warnings:
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "overall": overall,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def collect_evidence(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    if not root.is_dir():
        raise EvidenceCollectionError(
            f"프로젝트 루트가 아닙니다: {root}"
        )

    important_paths = inspect_important_paths(root)
    selected_artifacts = select_artifacts(root)

    evidence: dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": PROJECT_NAME,
            "name_ko": PROJECT_NAME_KO,
            "root": str(root),
        },
        "readme": inspect_readme(root),
        "important_paths": important_paths,
        "fastapi": inspect_fastapi_endpoints(root),
        "inventory": list_inventory(root),
        "selected_artifacts": selected_artifacts,
        "artifact_metrics": collect_artifact_metrics(
            root,
            selected_artifacts,
        ),
        "run_commands": build_run_commands(important_paths),
        "policy": {
            "readme_modified": False,
            "source_modified": False,
            "training_executed": False,
            "inference_executed": False,
            "new_dependency_added": False,
        },
    }
    evidence["status"] = derive_status(evidence)
    return evidence


def write_outputs(
    *,
    project_root: Path,
    evidence: dict[str, Any],
    json_output: Path,
    markdown_output: Path,
) -> tuple[Path, Path]:
    root = project_root.resolve()
    json_path = (
        json_output
        if json_output.is_absolute()
        else root / json_output
    )
    markdown_path = (
        markdown_output
        if markdown_output.is_absolute()
        else root / markdown_output
    )

    _write_text_atomically(
        json_path,
        json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write_text_atomically(
        markdown_path,
        build_markdown(evidence),
    )
    return json_path, markdown_path


def print_summary(
    evidence: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    status = evidence["status"]
    counts = evidence["inventory"]["counts"]

    print("=" * 100)
    print("DAY 14 - FINAL INTEGRATION EVIDENCE COLLECTION")
    print("=" * 100)
    print(f"Overall status            : {status['overall']}")
    print(f"README headings           : {evidence['readme']['heading_count']}")
    print(f"README markers            : {evidence['readme']['marker_count']}")
    print(f"Reports                   : {counts['reports']}")
    print(f"Artifacts                 : {counts['artifacts']}")
    print(f"Figures                   : {counts['figures']}")
    print(f"Source files              : {counts['source_files']}")
    print(f"Test files                : {counts['test_files']}")
    print(f"Script files              : {counts['script_files']}")
    print(
        "Expected endpoints        : "
        f"{sum(evidence['fastapi']['expected_status'].values())}"
        f"/{len(EXPECTED_ENDPOINTS)}"
    )
    print(f"Errors                    : {status['error_count']}")
    print(f"Warnings                  : {status['warning_count']}")

    for error in status["errors"]:
        print(f"[ERROR] {error}")
    for warning in status["warnings"]:
        print(f"[WARN] {warning}")

    print(f"[JSON] {json_path.resolve()}")
    print(f"[MARKDOWN] {markdown_path.resolve()}")
    print("[README] Not modified")
    print("[SOURCE] Not modified")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        evidence = collect_evidence(args.project_root)
        json_path, markdown_path = write_outputs(
            project_root=args.project_root,
            evidence=evidence,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )
    except EvidenceCollectionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print_summary(evidence, json_path, markdown_path)

    if (
        args.fail_on_error
        and evidence["status"]["overall"] == "FAIL"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

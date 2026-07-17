"""Day 14 final-integration prerequisite inspector.

이 스크립트는 README를 수정하지 않고 저장소의 현재 상태만 읽어서 점검한다.

주요 점검 항목
- README Day Marker 중복·누락
- Report·Artifact·Figure 인벤토리
- src·tests·scripts 트리
- FastAPI Endpoint 정적 수집
- Streamlit 페이지·서비스·모델·데이터셋 관련 파일 요약
- README 내부 상대 링크의 존재 여부
- 오래된 테스트 수와 과장되거나 혼동을 부를 수 있는 표현
- Day 12·Day 13 JSON Artifact의 핵심 수치
- 실행 명령에 필요한 파일·모듈의 정적 존재 여부

외부 패키지를 추가하지 않으며 Python 표준 라이브러리만 사용한다.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
INSPECTOR_RELATIVE_PATH = Path(
    "scripts/inspect_day14_final_integration_prerequisites.py"
)
CONTEXT_REBUILD_RELATIVE_PATH = Path(
    "scripts/rebuild_day13_api_core_context.py"
)
DAY14_DOCS_RELATIVE_PATH = Path(
    "scripts/create_day14_docs.py"
)
SELF_REFERENTIAL_QUALITY_EXCLUSIONS = (
    INSPECTOR_RELATIVE_PATH,
    CONTEXT_REBUILD_RELATIVE_PATH,
    DAY14_DOCS_RELATIVE_PATH,
)
DIAGNOSTIC_BACKUP_DIRECTORY = Path(
    "reports/artifacts/backups"
)
DEFAULT_OUTPUT = (
    Path("reports")
    / "artifacts"
    / "day14_final_integration_prerequisites_inspection.json"
)

DAY_MARKER_PATTERN = re.compile(
    r"<!--\s*(DAY(?P<day>\d+)_[A-Z0-9_]+)_(?P<kind>START|END)\s*-->",
    re.IGNORECASE,
)
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PYTEST_COUNT_PATTERN = re.compile(r"(?P<count>\d{1,6})\s+passed\b", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".ps1",
}

OVERCLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "production_validation_overclaim": (
        "실제 생산 환경에서 검증 완료",
        "산업 현장 배포 완료",
        "실시간 생산 시스템 구축",
        "production environment validated",
        "deployed to production",
    ),
    "metric_overclaim": (
        "COCO 공식 mAP와 완전히 동일",
        "official COCO mAP",
        "identical to COCOeval",
    ),
    "opencv_detection_confusion": (
        "OpenCV Contour가 실제 결함 위치",
        "Contour는 실제 결함 위치",
        "Contour is the ground truth",
        "Contour is a detection prediction",
    ),
    "prediction_ground_truth_confusion": (
        "Detection Prediction이 Ground Truth",
        "Prediction is ground truth",
        "prediction equals ground truth",
    ),
}

MOJIBAKE_TOKENS = (
    "\ufffd",  # Unicode replacement character
    "媛",
    "吏",
    "쒖",
    "湲",
    "怨",
)

EXPECTED_EXACT_FILES = (
    "README.md",
    "requirements.txt",
    ".gitignore",
    "src/dashboard/app.py",
    "src/dashboard/detection_api_client.py",
    "src/dashboard/detection_session_state.py",
    "src/dashboard/detection_ui_helpers.py",
    "src/dashboard/detection_page.py",
    "src/dashboard/pages/2_Detection.py",
    "reports/day13_detection_fastapi_streamlit_integration_summary.md",
    "reports/artifacts/day13_detection_api_stage1_inspection.json",
    "reports/artifacts/day13_detection_api_smoke_test.json",
    "reports/artifacts/day13_detection_dashboard_stage2_inspection.json",
    "reports/artifacts/day13_detection_dashboard_api_client_validation.json",
    "reports/artifacts/day13_detection_integration_summary.json",
    "reports/figures/day13_detection_dashboard_overlay.png",
    "models/detection/day12_detection_latest.pt",
    "models/detection/day12_detection_best.pt",
    "data/processed/neu_det/splits.json",
)

CANDIDATE_GROUPS: dict[str, tuple[str, ...]] = {
    "fastapi_app": (
        "src/api/app.py",
        "src/app.py",
        "app.py",
    ),
    "classification_service": (
        "src/api/services/classification_service.py",
        "src/services/classification_service.py",
        "src/inference/classification_service.py",
    ),
    "detection_service": (
        "src/api/services/detection_service.py",
        "src/services/detection_service.py",
        "src/inference/detection_service.py",
    ),
    "api_schema": (
        "src/api/schemas.py",
        "src/api/schema.py",
        "src/schemas.py",
    ),
    "opencv_pipeline": (
        "src/opencv/pipeline.py",
        "src/opencv_analysis/pipeline.py",
        "src/analysis/opencv_pipeline.py",
    ),
    "detection_model_factory": (
        "src/detection/model.py",
        "src/detection/model_factory.py",
        "src/models/detection.py",
    ),
}

STATIC_RUN_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "pytest": ("tests",),
    "fastapi_uvicorn": (
        "src/api/app.py",
        "src/app.py",
        "app.py",
    ),
    "streamlit_classification": ("src/dashboard/app.py",),
    "streamlit_detection_page": ("src/dashboard/pages/2_Detection.py",),
    "day14_inspector": (
        "scripts/inspect_day14_final_integration_prerequisites.py",
    ),
}


def _as_posix(path: Path) -> str:
    """Windows와 Linux에서 동일한 JSON 경로 표기를 사용한다."""

    return path.as_posix()


def _relative(path: Path, root: Path) -> str:
    """프로젝트 루트 기준 상대 경로를 반환한다."""

    return _as_posix(path.resolve().relative_to(root.resolve()))


def read_text_safely(path: Path) -> tuple[str, str, str | None]:
    """UTF-8을 우선하고 필요한 경우 CP949까지 시도한다.

    반환값:
        (텍스트, 사용한 인코딩, 오류 메시지)
    """

    encodings = ("utf-8-sig", "utf-8", "cp949")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding, None
        except (UnicodeDecodeError, OSError) as exc:
            last_error = exc

    return "", "unreadable", str(last_error) if last_error else "unknown error"


def safe_json_load(path: Path) -> tuple[Any | None, str | None]:
    """JSON을 읽고 실패 원인을 문자열로 반환한다."""

    text, _encoding, read_error = read_text_safely(path)
    if read_error:
        return None, read_error

    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} at line {exc.lineno}, column {exc.colno}"


def list_files(root: Path, directory: str) -> list[Path]:
    """특정 디렉터리 아래의 모든 파일을 정렬해 반환한다."""

    target = root / directory
    if not target.exists():
        return []
    return sorted(path for path in target.rglob("*") if path.is_file())


def build_tree(root: Path, directory: str, max_depth: int = 6) -> list[str]:
    """JSON에 저장할 간단한 파일 트리를 만든다."""

    target = root / directory
    if not target.exists():
        return []

    rows: list[str] = []
    for path in sorted(target.rglob("*")):
        relative = path.relative_to(target)
        if len(relative.parts) > max_depth:
            continue
        suffix = "/" if path.is_dir() else ""
        rows.append(f"{directory}/{_as_posix(relative)}{suffix}")
    return rows


def inspect_markers(readme_text: str) -> dict[str, Any]:
    """README의 Day Marker 개수와 중복·누락을 점검한다."""

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    days_found: set[int] = set()

    for match in DAY_MARKER_PATTERN.finditer(readme_text):
        marker_name = match.group(1).upper()
        marker_kind = match.group("kind").upper()
        grouped[marker_name][marker_kind] += 1
        days_found.add(int(match.group("day")))

    marker_details: dict[str, dict[str, int]] = {}
    invalid_pairs: list[dict[str, Any]] = []

    for marker_name in sorted(grouped):
        counts = grouped[marker_name]
        detail = {
            "start_count": counts["START"],
            "end_count": counts["END"],
        }
        marker_details[marker_name] = detail
        if detail["start_count"] != 1 or detail["end_count"] != 1:
            invalid_pairs.append({"marker": marker_name, **detail})

    missing_days = [day for day in range(1, 14) if day not in days_found]

    return {
        "marker_group_count": len(marker_details),
        "days_found": sorted(days_found),
        "missing_days_1_to_13": missing_days,
        "marker_details": marker_details,
        "invalid_or_duplicate_pairs": invalid_pairs,
        "all_day_1_to_13_present": not missing_days,
        "all_pairs_single": not invalid_pairs,
    }


def inspect_inventory(root: Path) -> dict[str, Any]:
    """Report·Artifact·Figure와 주요 소스 트리를 수집한다."""

    reports = list_files(root, "reports")
    report_markdown = [
        path for path in reports if path.suffix.lower() == ".md"
    ]
    artifacts = list_files(root, "reports/artifacts")
    figures = list_files(root, "reports/figures")

    day_report_coverage: dict[str, list[str]] = {}
    for day in range(1, 14):
        pattern = re.compile(rf"day0?{day}(?:_|-)", re.IGNORECASE)
        matches = [
            _relative(path, root)
            for path in report_markdown
            if pattern.search(path.name)
        ]
        day_report_coverage[f"day{day}"] = matches

    exact_file_status = {
        path_string: (root / path_string).is_file()
        for path_string in EXPECTED_EXACT_FILES
    }

    candidate_status: dict[str, Any] = {}
    for group_name, candidates in CANDIDATE_GROUPS.items():
        existing = [
            candidate
            for candidate in candidates
            if (root / candidate).is_file()
        ]
        candidate_status[group_name] = {
            "found": existing,
            "found_any": bool(existing),
            "candidates": list(candidates),
        }

    return {
        "counts": {
            "report_markdown": len(report_markdown),
            "artifacts": len(artifacts),
            "figures": len(figures),
            "src_files": len(list_files(root, "src")),
            "test_files": len(list_files(root, "tests")),
            "script_files": len(list_files(root, "scripts")),
        },
        "day_report_coverage": day_report_coverage,
        "missing_day_reports_1_to_13": [
            day for day, matches in day_report_coverage.items() if not matches
        ],
        "exact_file_status": exact_file_status,
        "missing_expected_exact_files": [
            path for path, exists in exact_file_status.items() if not exists
        ],
        "candidate_groups": candidate_status,
        "reports": [_relative(path, root) for path in report_markdown],
        "artifacts": [_relative(path, root) for path in artifacts],
        "figures": [_relative(path, root) for path in figures],
        "trees": {
            "src": build_tree(root, "src"),
            "tests": build_tree(root, "tests"),
            "scripts": build_tree(root, "scripts"),
        },
    }


def iter_text_files(root: Path, directories: Iterable[str]) -> Iterable[Path]:
    """점검 대상 디렉터리의 텍스트 파일을 순회한다."""

    seen: set[Path] = set()
    for directory in directories:
        target = root / directory
        if target.is_file():
            candidates = [target]
        elif target.exists():
            candidates = target.rglob("*")
        else:
            continue

        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path



def _evaluate_route_string(
    node: ast.AST,
    constants: dict[str, str],
) -> str | None:
    """AST 표현식에서 정적으로 계산 가능한 경로 문자열을 반환한다.

    지원 범위:
    - "/api/v1/health" 같은 문자열 리터럴
    - API_PREFIX 같은 같은 파일 내부 문자열 상수
    - API_PREFIX + "/health" 문자열 결합
    - f"{API_PREFIX}/health" 형태의 f-string
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.Name):
        return constants.get(node.id)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _evaluate_route_string(node.left, constants)
        right = _evaluate_route_string(node.right, constants)
        if left is not None and right is not None:
            return left + right
        return None

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue

            if isinstance(value, ast.FormattedValue):
                evaluated = _evaluate_route_string(value.value, constants)
                if evaluated is None:
                    return None
                parts.append(evaluated)
                continue

            return None
        return "".join(parts)

    return None


def _collect_module_string_constants(tree: ast.Module) -> dict[str, str]:
    """모듈 최상위의 문자열 상수를 반복적으로 계산한다."""

    constants: dict[str, str] = {}
    assignments: list[tuple[str, ast.AST]] = []

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, statement.value))
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            if statement.value is not None:
                assignments.append((statement.target.id, statement.value))

    unresolved = assignments[:]
    while unresolved:
        next_unresolved: list[tuple[str, ast.AST]] = []
        changed = False

        for name, expression in unresolved:
            value = _evaluate_route_string(expression, constants)
            if value is None:
                next_unresolved.append((name, expression))
                continue

            constants[name] = value
            changed = True

        if not changed:
            break
        unresolved = next_unresolved

    return constants


def inspect_endpoints(root: Path) -> dict[str, Any]:
    """Python AST에서 FastAPI HTTP decorator를 정적으로 수집한다.

    decorator 객체 이름을 ``app`` 또는 ``router``로 제한하지 않는다.
    따라서 ``api``, ``application``, ``prediction_router`` 같은 이름도
    동일하게 검사한다.
    """

    endpoints: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    unresolved_decorators: list[dict[str, Any]] = []

    for path in iter_text_files(root, ("src",)):
        if path.suffix.lower() != ".py":
            continue

        text, _encoding, error = read_text_safely(path)
        if error:
            parse_errors.append(
                {"file": _relative(path, root), "error": error}
            )
            continue

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            parse_errors.append(
                {
                    "file": _relative(path, root),
                    "error": (
                        f"{exc.msg} at line {exc.lineno}, "
                        f"column {exc.offset}"
                    ),
                }
            )
            continue

        constants = _collect_module_string_constants(tree)

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

                method = decorator.func.attr.casefold()
                if method not in HTTP_METHODS:
                    continue

                relative_file = _relative(path, root)
                if not decorator.args:
                    unresolved_decorators.append(
                        {
                            "method": method.upper(),
                            "file": relative_file,
                            "line": getattr(decorator, "lineno", None),
                            "reason": "route path argument is missing",
                        }
                    )
                    continue

                endpoint_path = _evaluate_route_string(
                    decorator.args[0],
                    constants,
                )
                if endpoint_path is None:
                    unresolved_decorators.append(
                        {
                            "method": method.upper(),
                            "file": relative_file,
                            "line": getattr(decorator, "lineno", None),
                            "reason": "route path could not be resolved statically",
                            "expression": ast.unparse(decorator.args[0]),
                        }
                    )
                    continue

                endpoints.append(
                    {
                        "method": method.upper(),
                        "path": endpoint_path,
                        "file": relative_file,
                        "line": getattr(decorator, "lineno", None),
                        "function": node.name,
                    }
                )

    unique: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    for endpoint in endpoints:
        key = (
            endpoint["method"],
            endpoint["path"],
            endpoint["file"],
            endpoint["line"],
        )
        unique[key] = endpoint

    endpoints = sorted(
        unique.values(),
        key=lambda item: (
            item["path"],
            item["method"],
            item["file"],
            item["line"] or 0,
        ),
    )
    endpoint_pairs = {(item["method"], item["path"]) for item in endpoints}
    expected = {
        ("GET", "/api/v1/health"),
        ("POST", "/api/v1/predictions"),
        ("POST", "/api/v1/detection/predictions"),
    }

    return {
        "inspection_method": "python_ast",
        "endpoints": endpoints,
        "expected_endpoint_status": {
            f"{method} {path}": (method, path) in endpoint_pairs
            for method, path in sorted(expected)
        },
        "missing_expected_endpoints": [
            f"{method} {path}"
            for method, path in sorted(expected)
            if (method, path) not in endpoint_pairs
        ],
        "unresolved_decorators": unresolved_decorators,
        "read_or_parse_errors": parse_errors,
    }


def inspect_python_syntax(root: Path) -> dict[str, Any]:
    """src·scripts·tests Python 파일의 문법을 import 없이 점검한다."""

    errors: list[dict[str, Any]] = []
    checked = 0

    for path in iter_text_files(root, ("src", "scripts", "tests")):
        if path.suffix.lower() != ".py":
            continue

        checked += 1
        text, _encoding, read_error = read_text_safely(path)
        if read_error:
            errors.append(
                {
                    "file": _relative(path, root),
                    "type": "read_error",
                    "message": read_error,
                }
            )
            continue

        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(
                {
                    "file": _relative(path, root),
                    "type": "syntax_error",
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                }
            )

    return {
        "checked_python_files": checked,
        "syntax_error_count": len(errors),
        "errors": errors,
    }


def inspect_readme_links(root: Path, readme_text: str) -> dict[str, Any]:
    """README 내부 로컬 상대 링크의 존재 여부를 점검한다."""

    checked: list[dict[str, Any]] = []
    broken: list[dict[str, str]] = []

    for raw_target in MARKDOWN_LINK_PATTERN.findall(readme_text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
            continue

        clean_target = target.split("#", 1)[0]
        if not clean_target:
            continue

        candidate = (root / clean_target).resolve()
        exists = candidate.exists()
        item = {"target": target, "exists": exists}
        checked.append(item)
        if not exists:
            broken.append({"target": target})

    return {
        "checked_relative_links": checked,
        "broken_relative_links": broken,
        "broken_count": len(broken),
    }


def inspect_readme_test_counts(readme_text: str) -> dict[str, Any]:
    """README에 기록된 pytest pass 수를 수집한다."""

    counts = [int(match.group("count")) for match in PYTEST_COUNT_PATTERN.finditer(readme_text)]
    return {
        "counts_found": counts,
        "maximum_count_found": max(counts) if counts else None,
        "contains_day13_final_count_1668": 1668 in counts,
    }



def _contains_mojibake(text: str) -> bool:
    """대표적인 문자 깨짐 토큰이 포함됐는지 반환한다."""

    return any(token in text for token in MOJIBAKE_TOKENS)


def _iter_user_facing_quality_files(root: Path) -> Iterable[Path]:
    """최종 README와 사용자에게 노출되는 문서·코드를 순회한다.

    ``reports/artifacts``의 원시 Context Dump와 생성 JSON은 실행 증거이지만
    최종 README 본문이 아니므로 사용자 문서 경고 집계에서 분리한다.
    """

    readme = root / "README.md"
    if readme.is_file():
        yield readme

    reports_root = root / "reports"
    if reports_root.exists():
        for path in sorted(reports_root.rglob("*.md")):
            if "artifacts" in path.relative_to(reports_root).parts:
                continue
            if path.is_file():
                yield path

    excluded_paths = {
        (root / relative_path).resolve()
        for relative_path in SELF_REFERENTIAL_QUALITY_EXCLUSIONS
    }

    for directory in ("src", "scripts"):
        target = root / directory
        if not target.exists():
            continue

        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() in excluded_paths:
                # 문자 품질 점검기와 복구 도구가 탐지용으로 보관한 토큰은
                # 사용자 문서 내용이 아니므로 자기 참조 점검에서 제외한다.
                continue
            if path.suffix.lower() not in {".py", ".md", ".txt", ".ps1"}:
                continue
            yield path


def inspect_diagnostic_artifact_text(root: Path) -> dict[str, Any]:
    """원시 진단 Artifact의 문자 깨짐을 정보 항목으로만 기록한다."""

    artifact_root = root / "reports" / "artifacts"
    backup_root = (root / DIAGNOSTIC_BACKUP_DIRECTORY).resolve()
    files: list[dict[str, Any]] = []
    excluded_backups: list[str] = []

    if artifact_root.exists():
        for path in sorted(artifact_root.rglob("*.txt")):
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(backup_root)
                excluded_backups.append(_relative(path, root))
                continue
            except ValueError:
                pass
            text, encoding, error = read_text_safely(path)
            if error:
                files.append(
                    {
                        "file": _relative(path, root),
                        "encoding": encoding,
                        "read_error": error,
                        "possible_mojibake": None,
                    }
                )
                continue

            if _contains_mojibake(text):
                files.append(
                    {
                        "file": _relative(path, root),
                        "encoding": encoding,
                        "read_error": None,
                        "possible_mojibake": True,
                    }
                )

    return {
        "files_with_possible_mojibake": files,
        "file_count": len(files),
        "excluded_backup_files": excluded_backups,
        "excluded_backup_count": len(excluded_backups),
        "severity": "INFO",
        "note": (
            "활성 원시 Context Dump는 README·보고서·실행 소스와 분리한다. "
            "복구 전 원본을 보존한 reports/artifacts/backups 아래 TXT는 "
            "의도적인 기록이므로 활성 진단 경고에서 제외한다."
        ),
    }


def find_suspicious_text(root: Path) -> dict[str, Any]:
    """사용자 노출 문서와 소스에서 깨진 한글·과장 표현을 찾는다."""

    findings: list[dict[str, Any]] = []
    scanned_files = 0

    for path in _iter_user_facing_quality_files(root):
        text, _encoding, error = read_text_safely(path)
        if error:
            findings.append(
                {
                    "file": _relative(path, root),
                    "category": "unreadable_text",
                    "line": None,
                    "matched": error,
                }
            )
            continue

        scanned_files += 1
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            for token in MOJIBAKE_TOKENS:
                if token in line:
                    findings.append(
                        {
                            "file": _relative(path, root),
                            "category": "possible_mojibake",
                            "line": line_number,
                            "matched": token,
                            "text": line.strip()[:240],
                        }
                    )

            lowered = line.casefold()
            for category, patterns in OVERCLAIM_PATTERNS.items():
                for pattern in patterns:
                    if pattern.casefold() in lowered:
                        findings.append(
                            {
                                "file": _relative(path, root),
                                "category": category,
                                "line": line_number,
                                "matched": pattern,
                                "text": line.strip()[:240],
                            }
                        )

    category_counts = Counter(item["category"] for item in findings)
    return {
        "scope": (
            "README.md, reports의 Markdown 보고서, src·scripts의 "
            "사용자 노출 텍스트 파일"
        ),
        "excluded_self_referential_files": [
            path.as_posix()
            for path in SELF_REFERENTIAL_QUALITY_EXCLUSIONS
        ],
        "scanned_files": scanned_files,
        "finding_count": len(findings),
        "category_counts": dict(sorted(category_counts.items())),
        "findings": findings,
        "diagnostic_artifacts": inspect_diagnostic_artifact_text(root),
    }


def recursively_find_keys(value: Any, key_names: set[str], prefix: str = "") -> list[dict[str, Any]]:
    """중첩 JSON에서 관심 키를 재귀적으로 찾는다."""

    found: list[dict[str, Any]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in key_names:
                found.append({"json_path": child_path, "value": child})
            found.extend(recursively_find_keys(child, key_names, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{prefix}[{index}]"
            found.extend(recursively_find_keys(child, key_names, child_path))

    return found


def inspect_key_artifacts(root: Path) -> dict[str, Any]:
    """Day 12·Day 13 핵심 Artifact를 읽고 주요 수치를 추출한다."""

    artifact_dir = root / "reports" / "artifacts"
    if not artifact_dir.exists():
        return {
            "artifact_directory_exists": False,
            "files": [],
        }

    name_tokens = ("day12", "day13")
    selected = [
        path
        for path in sorted(artifact_dir.glob("*.json"))
        if any(token in path.name.casefold() for token in name_tokens)
    ]

    interesting_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "map_50",
        "map@0.50",
        "project_map_50_95",
        "project_map@0.50:0.95",
        "test_images",
        "ground_truth_boxes",
        "tp",
        "fp",
        "fn",
        "checkpoint_epoch",
        "checkpoint_metric_name",
        "checkpoint_metric_value",
        "regression_test_count",
        "test_count",
        "warning_count",
        "runtime_seconds",
        "manual_browser_check_status",
    }

    rows: list[dict[str, Any]] = []
    for path in selected:
        data, error = safe_json_load(path)
        row: dict[str, Any] = {
            "file": _relative(path, root),
            "valid_json": error is None,
        }
        if error:
            row["error"] = error
        else:
            row["interesting_values"] = recursively_find_keys(
                data,
                {key.casefold() for key in interesting_keys},
            )
        rows.append(row)

    return {
        "artifact_directory_exists": True,
        "files": rows,
    }


def inspect_static_run_requirements(root: Path) -> dict[str, Any]:
    """대표 실행 명령에 필요한 경로가 존재하는지 정적으로 확인한다."""

    results: dict[str, Any] = {}

    for command_name, candidates in STATIC_RUN_REQUIREMENTS.items():
        existing = [
            candidate
            for candidate in candidates
            if (root / candidate).exists()
        ]
        results[command_name] = {
            "found_any": bool(existing),
            "existing_candidates": existing,
            "candidates": list(candidates),
        }

    return {
        "commands": results,
        "missing_command_requirements": [
            command_name
            for command_name, result in results.items()
            if not result["found_any"]
        ],
        "note": (
            "정적 존재 여부만 점검한다. 실제 FastAPI·Streamlit 실행과 "
            "Checkpoint 추론 검증은 별도 명령으로 수행해야 한다."
        ),
    }


def inspect_dashboard_boundaries(root: Path) -> dict[str, Any]:
    """Detection Streamlit이 모델을 직접 로딩하는지 정적으로 경고한다."""

    target_files = (
        "src/dashboard/detection_page.py",
        "src/dashboard/pages/2_Detection.py",
        "src/dashboard/detection_api_client.py",
    )
    forbidden_patterns = {
        "torch_import": re.compile(r"^\s*(?:from\s+torch|import\s+torch)\b", re.MULTILINE),
        "torchvision_import": re.compile(
            r"^\s*(?:from\s+torchvision|import\s+torchvision)\b",
            re.MULTILINE,
        ),
        "checkpoint_load": re.compile(r"\btorch\s*\.\s*load\s*\("),
        "direct_model_factory_hint": re.compile(
            r"\b(?:create|build|get)_detection_model\s*\(",
            re.IGNORECASE,
        ),
    }

    findings: list[dict[str, str]] = []
    status: dict[str, bool] = {}

    for relative_path in target_files:
        path = root / relative_path
        status[relative_path] = path.is_file()
        if not path.is_file():
            continue

        text, _encoding, error = read_text_safely(path)
        if error:
            findings.append(
                {
                    "file": relative_path,
                    "category": "read_error",
                    "matched": error,
                }
            )
            continue

        for category, pattern in forbidden_patterns.items():
            if pattern.search(text):
                findings.append(
                    {
                        "file": relative_path,
                        "category": category,
                        "matched": pattern.pattern,
                    }
                )

    return {
        "target_file_status": status,
        "direct_model_access_findings": findings,
        "api_client_only_static_check_passed": not findings,
    }



def derive_status(result: dict[str, Any]) -> dict[str, Any]:
    """치명적 오류, 경고, 정보 항목을 분리한다."""

    errors: list[str] = []
    warnings: list[str] = []
    information: list[str] = []

    if not result["project"]["readme_exists"]:
        errors.append("README.md가 없습니다.")

    syntax = result["python_syntax"]
    if syntax["syntax_error_count"]:
        errors.append(
            f"Python 문법 오류가 {syntax['syntax_error_count']}개 있습니다."
        )

    markers = result["readme"]["markers"]
    if markers["invalid_or_duplicate_pairs"]:
        errors.append("README Day Marker에 중복 또는 START/END 불일치가 있습니다.")
    if markers["missing_days_1_to_13"]:
        information.append(
            "README에 Marker가 없는 기존 Day가 있습니다: "
            + ", ".join(
                f"Day {day}" for day in markers["missing_days_1_to_13"]
            )
            + ". Marker 부재만으로 본문 누락을 의미하지 않습니다."
        )

    endpoints = result["fastapi"]
    if endpoints["missing_expected_endpoints"]:
        warnings.append(
            "예상 FastAPI Endpoint 일부를 Python AST 점검에서 찾지 못했습니다."
        )
    if endpoints["unresolved_decorators"]:
        information.append(
            "경로를 정적으로 계산할 수 없는 FastAPI decorator가 "
            f"{len(endpoints['unresolved_decorators'])}개 있습니다."
        )

    inventory = result["inventory"]
    if inventory["missing_expected_exact_files"]:
        warnings.append(
            "사용자 제공 Day 13 기준의 예상 파일 일부가 현재 저장소에 없습니다."
        )
    if inventory["missing_day_reports_1_to_13"]:
        warnings.append("Day 1~13 보고서 이름 패턴의 누락 가능성이 있습니다.")

    links = result["readme"]["relative_links"]
    if links["broken_count"]:
        warnings.append(
            f"README 상대 링크 {links['broken_count']}개가 존재하지 않습니다."
        )

    suspicious = result["text_quality"]
    if suspicious["finding_count"]:
        warnings.append(
            "README·Markdown 보고서·실행 소스에서 "
            f"깨진 한글·과장·역할 혼동 후보가 "
            f"{suspicious['finding_count']}개 있습니다."
        )

    diagnostic_count = suspicious["diagnostic_artifacts"]["file_count"]
    if diagnostic_count:
        information.append(
            "문자 깨짐 가능성이 있는 원시 진단 TXT Artifact가 "
            f"{diagnostic_count}개 있습니다. 사용자 노출 문서 경고와 분리했습니다."
        )

    boundary = result["dashboard_boundary"]
    if boundary["direct_model_access_findings"]:
        errors.append(
            "Detection Dashboard에서 직접 모델 접근 가능성이 발견됐습니다."
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
        "information": information,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "information_count": len(information),
    }


def inspect_project(project_root: Path) -> dict[str, Any]:
    """전체 점검을 실행하고 직렬화 가능한 dict를 반환한다."""

    root = project_root.resolve()
    readme_path = root / "README.md"
    readme_text = ""
    readme_encoding = None
    readme_error = None

    if readme_path.is_file():
        readme_text, readme_encoding, readme_error = read_text_safely(readme_path)

    result: dict[str, Any] = {
        "schema_version": 5,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": PROJECT_NAME,
            "project_root": str(root),
            "readme_exists": readme_path.is_file(),
            "requirements_exists": (root / "requirements.txt").is_file(),
            "gitignore_exists": (root / ".gitignore").is_file(),
        },
        "readme": {
            "encoding": readme_encoding,
            "read_error": readme_error,
            "markers": inspect_markers(readme_text),
            "pytest_counts": inspect_readme_test_counts(readme_text),
            "relative_links": inspect_readme_links(root, readme_text),
        },
        "inventory": inspect_inventory(root),
        "fastapi": inspect_endpoints(root),
        "python_syntax": inspect_python_syntax(root),
        "dashboard_boundary": inspect_dashboard_boundaries(root),
        "key_artifacts": inspect_key_artifacts(root),
        "static_run_requirements": inspect_static_run_requirements(root),
        "text_quality": find_suspicious_text(root),
        "manual_browser_check": {
            "day13_status": "not_recorded",
            "day14_policy": (
                "자동 HTTP·API Client·Overlay 검증과 브라우저 수동 확인을 "
                "분리해서 기록한다. 수동 브라우저 확인 전에는 완료로 변경하지 않는다."
            ),
        },
        "write_policy": {
            "readme_modified": False,
            "source_modified": False,
            "inspection_only": True,
        },
    }
    result["status"] = derive_status(result)
    return result


def write_json(result: dict[str, Any], output_path: Path) -> None:
    """부모 디렉터리를 만든 뒤 UTF-8 JSON Artifact를 저장한다."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    """PowerShell에서 빠르게 확인할 핵심 결과를 출력한다."""

    status = result["status"]
    inventory = result["inventory"]["counts"]
    markers = result["readme"]["markers"]
    endpoint_status = result["fastapi"]["expected_endpoint_status"]

    print("=" * 100)
    print("DAY 14 - FINAL INTEGRATION PREREQUISITES INSPECTION")
    print("=" * 100)
    print(f"Project root                 : {result['project']['project_root']}")
    print(f"Overall status               : {status['overall']}")
    print(f"README exists                : {result['project']['readme_exists']}")
    print(f"README marker groups         : {markers['marker_group_count']}")
    print(f"Missing Day markers 1~13     : {markers['missing_days_1_to_13']}")
    print(
        "Invalid/duplicate markers    : "
        f"{len(markers['invalid_or_duplicate_pairs'])}"
    )
    print(f"Report markdown files        : {inventory['report_markdown']}")
    print(f"Artifact files               : {inventory['artifacts']}")
    print(f"Figure files                 : {inventory['figures']}")
    print(f"Source files                 : {inventory['src_files']}")
    print(f"Test files                   : {inventory['test_files']}")
    print(f"Script files                 : {inventory['script_files']}")
    print(
        "Python syntax errors         : "
        f"{result['python_syntax']['syntax_error_count']}"
    )

    for endpoint_name, exists in endpoint_status.items():
        print(f"Endpoint {endpoint_name:<39}: {'PASS' if exists else 'NOT FOUND'}")

    print(
        "Dashboard API-client boundary: "
        f"{'PASS' if result['dashboard_boundary']['api_client_only_static_check_passed'] else 'FAIL'}"
    )
    print(
        "Broken README links          : "
        f"{result['readme']['relative_links']['broken_count']}"
    )
    print(
        "Text-quality findings        : "
        f"{result['text_quality']['finding_count']}"
    )
    print(f"Errors                       : {status['error_count']}")
    print(f"Warnings                     : {status['warning_count']}")
    print(f"Information                  : {status['information_count']}")

    for error in status["errors"]:
        print(f"[ERROR] {error}")
    for warning in status["warnings"]:
        print(f"[WARN] {warning}")
    for information in status["information"]:
        print(f"[INFO] {information}")

    print(f"[ARTIFACT] {output_path.resolve()}")
    print("[README] Not modified")
    print("[SOURCE] Not modified")


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자를 정의한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect Day 14 final-integration prerequisites without changing "
            "README or existing application code."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="프로젝트 루트 경로. 기본값은 현재 디렉터리다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "JSON Artifact 경로. 상대 경로이면 project-root 기준으로 해석한다."
        ),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="점검 결과가 FAIL이면 종료 코드 1을 반환한다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""

    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    if not project_root.exists() or not project_root.is_dir():
        parser.error(f"project root does not exist or is not a directory: {project_root}")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path

    result = inspect_project(project_root)
    write_json(result, output_path)
    print_summary(result, output_path)

    if args.fail_on_error and result["status"]["overall"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


TEXT_EXTENSIONS = {
    ".py", ".toml", ".txt", ".md", ".yaml", ".yml", ".json", ".ini", ".cfg",
}

SKIP_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "models",
    "node_modules",
    "reports",
}

RELEVANT_NAME_KEYWORDS = (
    "api",
    "app",
    "client",
    "config",
    "dashboard",
    "dependency",
    "detection",
    "exception",
    "health",
    "image",
    "inference",
    "main",
    "opencv",
    "prediction",
    "router",
    "schema",
    "service",
    "session",
    "streamlit",
    "ui",
)

ROUTE_DECORATOR_PATTERN = re.compile(
    r"@\s*(?P<object>[A-Za-z_][\w\.]*)\."
    r"(?P<method>get|post|put|patch|delete|options|head)\s*"
    r"\(\s*(?P<quote>['\"])(?P<path>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)

INCLUDE_ROUTER_PATTERN = re.compile(
    r"include_router\s*\((?P<body>.*?)\)",
    re.IGNORECASE | re.DOTALL,
)

EXCEPTION_HANDLER_PATTERN = re.compile(
    r"@\s*(?P<object>[A-Za-z_][\w\.]*)\.exception_handler\s*\((?P<target>.*?)\)",
    re.IGNORECASE | re.DOTALL,
)

STREAMLIT_CALL_PATTERN = re.compile(
    r"\bst\.(?P<call>"
    r"file_uploader|slider|number_input|dataframe|table|image|error|warning|info|success|"
    r"session_state|set_page_config|tabs|columns|metric|button"
    r")\b"
)

IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+(?P<from_module>[\w\.]+)\s+import|import\s+(?P<module>[\w\.]+))",
    re.MULTILINE,
)

KEY_STRING_PATTERN = re.compile(
    r"(?P<quote>['\"])(?P<value>"
    r"/api/v1/[^'\"]+|"
    r"day12_detection_best\.pt|"
    r"DAY12_DETECTION_[A-Z_]+|"
    r"multipart/form-data|"
    r"UploadFile|"
    r"HTTPException|"
    r"TestClient|"
    r"httpx\.(?:Client|AsyncClient)"
    r")(?P=quote)"
)


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    modified_time_ns: int
    role_hints: list[str]
    imports: list[str]
    routes: list[dict[str, str]]
    include_router_calls: list[str]
    exception_handlers: list[str]
    pydantic_models: list[str]
    functions: list[str]
    classes: list[str]
    streamlit_calls: list[str]
    key_strings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the existing repository before implementing Day 13 Detection "
            "FastAPI and Streamlit integration."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("reports/artifacts/day13_integration_prerequisites.json"),
        help="JSON artifact path relative to project root.",
    )
    parser.add_argument(
        "--run-collect-only",
        action="store_true",
        help="Run pytest --collect-only -q and store its result. No tests are executed.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("models/detection/day12_detection_best.pt"),
        help="Detection checkpoint path relative to project root.",
    )
    return parser.parse_args()


def resolve_inside_root(project_root: Path, path: Path) -> Path:
    project_root = project_root.resolve()
    resolved = path if path.is_absolute() else project_root / path
    return resolved.resolve()


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def iter_text_files(project_root: Path) -> Iterable[Path]:
    candidate_roots = [
        project_root / "src",
        project_root / "tests",
        project_root / "scripts",
        project_root / "dashboard",
        project_root / "streamlit",
        project_root / "app",
    ]

    root_files = [
        project_root / "README.md",
        project_root / "requirements.txt",
        project_root / "requirements-dev.txt",
        project_root / "pyproject.toml",
        project_root / "setup.cfg",
        project_root / "pytest.ini",
    ]

    seen: set[Path] = set()

    for file_path in root_files:
        if file_path.is_file():
            resolved = file_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    for candidate_root in candidate_roots:
        if not candidate_root.is_dir():
            continue

        for current_root, dirnames, filenames in os.walk(candidate_root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_DIRECTORY_NAMES and not name.endswith("_candidate")
            ]
            current_path = Path(current_root)
            for filename in filenames:
                path = current_path / filename
                if path.suffix.lower() not in TEXT_EXTENSIONS:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def role_hints_for(path: Path, text: str) -> list[str]:
    lowered_name = path.name.lower()
    lowered_path = path.as_posix().lower()
    hints: set[str] = set()

    mapping = {
        "fastapi_app": ("fastapi(", "include_router(", "uvicorn"),
        "router": ("apirouter(", "@router.", "include_router("),
        "schema": ("basemodel", "field(", "model_validator", "field_validator"),
        "model_service": ("load_state_dict", "torch.load", "inference_mode", ".eval()"),
        "image_validation": ("uploadfile", "pil.image", "image.verify", "mime", "content_type"),
        "exception_handler": ("exception_handler(", "httpexception"),
        "api_test": ("testclient", "httpx.client", "httpx.asyncclient"),
        "streamlit_page": ("import streamlit", "st.file_uploader", "st.set_page_config"),
        "api_client": ("httpx", "requests.post", "multipart"),
        "session_state": ("session_state",),
        "opencv": ("cv2.", "opencv", "contour", "morphology"),
        "detection": ("fasterrcnn", "bounding box", "detections", "/api/v1/detection", "day12_detection_best.pt"),
        "classification": ("/api/v1/predictions", "resnet18", "classification"),
    }

    combined = text.lower()
    for role, markers in mapping.items():
        if any(marker in combined for marker in markers):
            hints.add(role)

    if any(keyword in lowered_name or keyword in lowered_path for keyword in RELEVANT_NAME_KEYWORDS):
        hints.add("relevant_filename")

    return sorted(hints)


def parse_python_ast(path: Path, text: str) -> tuple[list[str], list[str], list[str]]:
    if path.suffix.lower() != ".py":
        return [], [], []

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return [], [], []

    functions: list[str] = []
    classes: list[str] = []
    pydantic_models: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            base_names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)
            if "BaseModel" in base_names:
                pydantic_models.append(node.name)

    return (
        sorted(set(functions)),
        sorted(set(classes)),
        sorted(set(pydantic_models)),
    )


def inspect_file(path: Path, project_root: Path) -> FileRecord:
    text = read_text(path)
    functions, classes, pydantic_models = parse_python_ast(path, text)

    routes = [
        {
            "object": match.group("object"),
            "method": match.group("method").upper(),
            "path": match.group("path"),
        }
        for match in ROUTE_DECORATOR_PATTERN.finditer(text)
    ]

    include_router_calls = [
        " ".join(match.group("body").split())[:500]
        for match in INCLUDE_ROUTER_PATTERN.finditer(text)
    ]

    exception_handlers = [
        " ".join(match.group("target").split())[:300]
        for match in EXCEPTION_HANDLER_PATTERN.finditer(text)
    ]

    imports = sorted(
        {
            match.group("from_module") or match.group("module")
            for match in IMPORT_PATTERN.finditer(text)
            if match.group("from_module") or match.group("module")
        }
    )

    streamlit_calls = sorted(
        {match.group("call") for match in STREAMLIT_CALL_PATTERN.finditer(text)}
    )

    key_strings = sorted(
        {match.group("value") for match in KEY_STRING_PATTERN.finditer(text)}
    )

    stat = path.stat()
    return FileRecord(
        path=safe_relative(path, project_root),
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
        role_hints=role_hints_for(path, text),
        imports=imports,
        routes=routes,
        include_router_calls=include_router_calls,
        exception_handlers=exception_handlers,
        pydantic_models=pydantic_models,
        functions=functions,
        classes=classes,
        streamlit_calls=streamlit_calls,
        key_strings=key_strings,
    )


def package_versions() -> dict[str, str | None]:
    packages = [
        "fastapi",
        "starlette",
        "pydantic",
        "python-multipart",
        "httpx",
        "streamlit",
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "Pillow",
        "matplotlib",
        "pytest",
        "uvicorn",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_or_simple(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        converted = [scalar_or_simple(item) for item in value[:50]]
        return converted
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 100:
                result["__truncated__"] = True
                break
            if isinstance(item, (str, bool, int, float, type(None), Path, list, tuple, dict)):
                result[str(key)] = scalar_or_simple(item)
            else:
                result[str(key)] = f"<{type(item).__name__}>"
        return result
    return f"<{type(value).__name__}>"


def inspect_checkpoint(path: Path, project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": safe_relative(path, project_root),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result

    stat = path.stat()
    result.update(
        {
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "sha256": sha256_file(path),
        }
    )

    try:
        import torch

        checkpoint = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        result["torch_load_status"] = "PASS"
        result["checkpoint_type"] = type(checkpoint).__name__

        if isinstance(checkpoint, dict):
            result["top_level_keys"] = sorted(str(key) for key in checkpoint.keys())
            metadata_keys = (
                "epoch",
                "best_epoch",
                "architecture",
                "model_name",
                "num_classes",
                "class_names",
                "class_mapping",
                "label_mapping",
                "best_metric",
                "best_metric_name",
                "best_metric_value",
                "validation_metrics",
                "metrics",
                "training_config",
                "model_config",
            )
            result["selected_metadata"] = {
                key: scalar_or_simple(checkpoint[key])
                for key in metadata_keys
                if key in checkpoint
            }

            state_dict_key = next(
                (
                    key
                    for key in ("model_state_dict", "state_dict", "model")
                    if key in checkpoint and isinstance(checkpoint[key], dict)
                ),
                None,
            )
            result["state_dict_key"] = state_dict_key
            if state_dict_key is not None:
                state_dict = checkpoint[state_dict_key]
                result["state_dict_parameter_count"] = len(state_dict)
                result["state_dict_sample_keys"] = list(state_dict.keys())[:25]
        del checkpoint
    except Exception as error:  # noqa: BLE001 - diagnostic script must preserve the error
        result["torch_load_status"] = "FAIL"
        result["torch_load_error_type"] = type(error).__name__
        result["torch_load_error"] = str(error)

    return result


def read_readme_status(project_root: Path) -> dict[str, Any]:
    readme_path = project_root / "README.md"
    result: dict[str, Any] = {
        "path": "README.md",
        "exists": readme_path.is_file(),
    }
    if not readme_path.is_file():
        return result

    text = read_text(readme_path)
    start_marker = "<!-- DAY12_DETECTION_TRAINING_EVALUATION_START -->"
    end_marker = "<!-- DAY12_DETECTION_TRAINING_EVALUATION_END -->"

    result.update(
        {
            "size_bytes": readme_path.stat().st_size,
            "line_count": len(text.splitlines()),
            "day12_start_marker_count": text.count(start_marker),
            "day12_end_marker_count": text.count(end_marker),
            "has_day13_marker": "DAY13_" in text,
        }
    )
    return result


def pytest_cache_status(project_root: Path) -> dict[str, Any]:
    cache_root = project_root / ".pytest_cache" / "v" / "cache"
    nodeids_path = cache_root / "nodeids"
    lastfailed_path = cache_root / "lastfailed"

    result: dict[str, Any] = {
        "cache_exists": cache_root.is_dir(),
        "nodeids_path": safe_relative(nodeids_path, project_root),
        "lastfailed_path": safe_relative(lastfailed_path, project_root),
    }

    if nodeids_path.is_file():
        try:
            nodeids = json.loads(read_text(nodeids_path))
            result["cached_collected_test_count"] = len(nodeids) if isinstance(nodeids, list) else None
        except (json.JSONDecodeError, OSError):
            result["cached_collected_test_count"] = None
    else:
        result["cached_collected_test_count"] = None

    if lastfailed_path.is_file():
        try:
            lastfailed = json.loads(read_text(lastfailed_path))
            result["cached_lastfailed_count"] = len(lastfailed) if isinstance(lastfailed, dict) else None
            result["cached_lastfailed_nodeids"] = list(lastfailed.keys())[:50] if isinstance(lastfailed, dict) else []
        except (json.JSONDecodeError, OSError):
            result["cached_lastfailed_count"] = None
            result["cached_lastfailed_nodeids"] = []
    else:
        result["cached_lastfailed_count"] = 0
        result["cached_lastfailed_nodeids"] = []

    return result


def run_pytest_collect_only(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-10000:],
        "stderr_tail": completed.stderr[-10000:],
    }


def build_summary(records: list[FileRecord]) -> dict[str, Any]:
    by_role: dict[str, list[str]] = {}
    endpoints: list[dict[str, str]] = []
    pydantic_models: list[dict[str, Any]] = []
    include_router_calls: list[dict[str, Any]] = []
    exception_handlers: list[dict[str, Any]] = []
    streamlit_files: list[dict[str, Any]] = []

    for record in records:
        for role in record.role_hints:
            by_role.setdefault(role, []).append(record.path)

        for route in record.routes:
            endpoints.append({"file": record.path, **route})

        if record.pydantic_models:
            pydantic_models.append(
                {"file": record.path, "models": record.pydantic_models}
            )

        if record.include_router_calls:
            include_router_calls.append(
                {"file": record.path, "calls": record.include_router_calls}
            )

        if record.exception_handlers:
            exception_handlers.append(
                {"file": record.path, "handlers": record.exception_handlers}
            )

        if record.streamlit_calls:
            streamlit_files.append(
                {"file": record.path, "calls": record.streamlit_calls}
            )

    for paths in by_role.values():
        paths.sort()

    endpoints.sort(key=lambda item: (item["path"], item["method"], item["file"]))

    return {
        "files_by_role": by_role,
        "detected_endpoints": endpoints,
        "pydantic_models": pydantic_models,
        "include_router_calls": include_router_calls,
        "exception_handlers": exception_handlers,
        "streamlit_files": streamlit_files,
    }


def validate_expected_items(
    project_root: Path,
    checkpoint: dict[str, Any],
    readme: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    detected_paths = {item["path"] for item in summary["detected_endpoints"]}

    checks = {
        "project_root_exists": project_root.is_dir(),
        "src_exists": (project_root / "src").is_dir(),
        "tests_exists": (project_root / "tests").is_dir(),
        "classification_health_endpoint_detected": "/api/v1/health" in detected_paths,
        "classification_prediction_endpoint_detected": "/api/v1/predictions" in detected_paths,
        "checkpoint_exists": bool(checkpoint.get("exists")),
        "checkpoint_torch_load_pass": checkpoint.get("torch_load_status") == "PASS",
        "readme_day12_start_marker_once": readme.get("day12_start_marker_count") == 1,
        "readme_day12_end_marker_once": readme.get("day12_end_marker_count") == 1,
        "fastapi_structure_detected": bool(
            summary["files_by_role"].get("fastapi_app")
            or summary["files_by_role"].get("router")
        ),
        "streamlit_structure_detected": bool(
            summary["files_by_role"].get("streamlit_page")
        ),
        "api_client_detected": bool(summary["files_by_role"].get("api_client")),
        "opencv_structure_detected": bool(summary["files_by_role"].get("opencv")),
    }

    return {
        "checks": checks,
        "pass_count": sum(bool(value) for value in checks.values()),
        "total_count": len(checks),
        "all_passed": all(bool(value) for value in checks.values()),
    }


def print_console_report(report: dict[str, Any], artifact_path: Path) -> None:
    environment = report["environment"]
    checkpoint = report["checkpoint"]
    readme = report["readme"]
    validation = report["validation"]
    summary = report["summary"]

    print("=" * 100)
    print("DAY 13 - DETECTION FASTAPI AND STREAMLIT INTEGRATION PREREQUISITES")
    print("=" * 100)
    print(f"Project root             : {report['project_root']}")
    print(f"Python executable        : {environment['python_executable']}")
    print(f"Python version           : {environment['python_version']}")
    print(f"Platform                 : {environment['platform']}")
    print(f"Inspected text files     : {len(report['files'])}")
    print(f"Detected endpoints       : {len(summary['detected_endpoints'])}")
    print(f"Checkpoint exists        : {checkpoint.get('exists')}")
    print(f"Checkpoint size (MB)     : {checkpoint.get('size_mb', 'N/A')}")
    print(f"Checkpoint torch.load    : {checkpoint.get('torch_load_status', 'NOT_RUN')}")
    print(f"README Day 12 start      : {readme.get('day12_start_marker_count', 'N/A')}")
    print(f"README Day 12 end        : {readme.get('day12_end_marker_count', 'N/A')}")
    print(f"Prerequisite checks      : {validation['pass_count']} / {validation['total_count']}")
    print(f"Artifact                 : {artifact_path}")
    print("-" * 100)

    if summary["detected_endpoints"]:
        print("[DETECTED ENDPOINTS]")
        for item in summary["detected_endpoints"]:
            print(
                f"{item['method']:6s} {item['path']:<45s} "
                f"({item['file']} -> {item['object']})"
            )
    else:
        print("[DETECTED ENDPOINTS] none")

    print("-" * 100)
    print("[PREREQUISITE CHECKS]")
    for name, passed in validation["checks"].items():
        print(f"[{'PASS' if passed else 'CHECK'}] {name}")

    print("=" * 100)
    print(
        "Next input: send this console output and "
        "reports/artifacts/day13_integration_prerequisites.json"
    )
    print("=" * 100)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    if not project_root.is_dir():
        print(f"[FAIL] Project root not found: {project_root}", file=sys.stderr)
        return 2

    artifact_path = resolve_inside_root(project_root, args.artifact_path)
    checkpoint_path = resolve_inside_root(project_root, args.checkpoint_path)

    records = [
        inspect_file(path, project_root)
        for path in sorted(iter_text_files(project_root))
    ]

    summary = build_summary(records)
    checkpoint = inspect_checkpoint(checkpoint_path, project_root)
    readme = read_readme_status(project_root)
    pytest_cache = pytest_cache_status(project_root)

    report: dict[str, Any] = {
        "report_name": "day13_integration_prerequisites",
        "project_root": str(project_root),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "checkpoint": checkpoint,
        "readme": readme,
        "pytest_cache": pytest_cache,
        "files": [asdict(record) for record in records],
        "summary": summary,
    }

    if args.run_collect_only:
        report["pytest_collect_only"] = run_pytest_collect_only(project_root)

    report["validation"] = validate_expected_items(
        project_root=project_root,
        checkpoint=checkpoint,
        readme=readme,
        summary=summary,
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_console_report(report, artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

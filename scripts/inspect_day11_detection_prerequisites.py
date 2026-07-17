"""Day 11 구현 전 실제 저장소·Artifact·Torchvision Detection API를 점검한다.

이 Script는 파일을 수정하지 않고 JSON Schema와 기존 Symbol을 읽기 전용으로 출력한다.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    Path("src/detection"),
    Path("src/detection/dataset_config.py"),
    Path("src/detection/annotation_parser.py"),
    Path("data/processed/neu_det/splits.json"),
    Path("reports/artifacts/day9_object_detection_dataset_analysis.json"),
    Path("reports/artifacts/day9_object_detection_dataset_split.json"),
    Path("reports/artifacts/day9_detection_visual_validation.json"),
    Path("reports/day9_object_detection_dataset_analysis_summary.md"),
    Path("reports/artifacts/day10_opencv_image_analysis.json"),
    Path("reports/artifacts/day10_opencv_visual_validation.json"),
    Path("reports/day10_opencv_image_analysis_pipeline_summary.md"),
    Path("requirements.txt"),
)

DETECTION_MODEL_NAMES = (
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_resnet50_fpn_v2",
)
WEIGHT_ENUM_NAMES = (
    "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "FasterRCNN_ResNet50_FPN_V2_Weights",
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_schema_preview(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    try:
        payload = json.loads(read_text(path))
    except (json.JSONDecodeError, OSError) as error:
        return {"exists": True, "error": repr(error)}

    preview: dict[str, Any] = {
        "exists": True,
        "top_level_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        preview["top_level_keys"] = sorted(payload.keys())
        for key, value in payload.items():
            if isinstance(value, list):
                preview[f"{key}_count"] = len(value)
                if value:
                    preview[f"{key}_first_item"] = value[0]
            elif isinstance(value, dict):
                preview[f"{key}_keys"] = sorted(value.keys())
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, list):
                        preview[f"{key}.{nested_key}_count"] = len(nested_value)
                        if nested_value:
                            preview[f"{key}.{nested_key}_first_item"] = nested_value[0]
    return preview


def iter_nested(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from iter_nested(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_nested(nested)


def find_duplicate_box_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError:
        return []

    matches: list[dict[str, Any]] = []
    for value in iter_nested(payload):
        if not isinstance(value, dict):
            continue
        code = str(value.get("code", "")).lower()
        joined_keys = " ".join(str(key).lower() for key in value.keys())
        if code == "duplicate_box" or "duplicate_box" in joined_keys:
            matches.append(value)
    return matches


def python_symbols(path: Path) -> dict[str, Any]:
    text = read_text(path)
    if not text:
        return {"exists": path.is_file(), "symbols": []}
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return {"exists": True, "parse_error": repr(error), "symbols": []}

    symbols: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "kind": type(node).__name__,
                    "name": node.name,
                    "line": node.lineno,
                }
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.append(
                        {
                            "kind": type(node).__name__,
                            "name": target.id,
                            "line": node.lineno,
                        }
                    )
    return {"exists": True, "symbols": symbols}


def inspect(project_root: Path) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    checks: list[CheckResult] = []

    for relative in REQUIRED_PATHS:
        path = root / relative
        checks.append(
            CheckResult(
                str(relative),
                "PASS" if path.exists() else "FAIL",
                "exists" if path.exists() else "not found",
            )
        )

    package_versions = {
        name: package_version(name)
        for name in (
            "torch",
            "torchvision",
            "Pillow",
            "numpy",
            "opencv-python",
        )
    }

    import_results: dict[str, str] = {}
    for module_name in ("torch", "torchvision", "PIL", "numpy", "cv2"):
        try:
            module = importlib.import_module(module_name)
            import_results[module_name] = str(
                getattr(module, "__version__", "unknown")
            )
        except Exception as error:  # pragma: no cover - 실제 환경 진단
            import_results[module_name] = f"IMPORT FAILED: {error!r}"

    model_api: dict[str, Any] = {}
    try:
        detection = importlib.import_module("torchvision.models.detection")
        for name in DETECTION_MODEL_NAMES:
            model_api[name] = {
                "available": callable(getattr(detection, name, None)),
            }
        for name in WEIGHT_ENUM_NAMES:
            enum = getattr(detection, name, None)
            model_api[name] = {
                "available": enum is not None,
                "default": repr(getattr(enum, "DEFAULT", None)) if enum else None,
            }
    except Exception as error:  # pragma: no cover
        model_api["import_error"] = repr(error)

    cache_root: Path | None = None
    cache_files: list[dict[str, Any]] = []
    try:
        torch_module = importlib.import_module("torch")
        cache_root = Path(torch_module.hub.get_dir()) / "checkpoints"
        if cache_root.is_dir():
            cache_files = [
                {
                    "name": path.name,
                    "size_mib": round(path.stat().st_size / (1024**2), 2),
                }
                for path in sorted(cache_root.iterdir())
                if path.is_file()
            ]
    except Exception as error:  # pragma: no cover
        cache_files = [{"error": repr(error)}]

    disk = shutil.disk_usage(root)
    detection_tree = [
        path.relative_to(root).as_posix()
        for path in sorted((root / "src" / "detection").rglob("*"))
        if path.is_file()
    ] if (root / "src" / "detection").is_dir() else []

    existing_dataset_model_code = [
        path
        for path in detection_tree
        if any(
            token in Path(path).name.lower()
            for token in ("dataset", "loader", "transform", "model", "factory")
        )
    ]

    json_paths = (
        Path("data/processed/neu_det/splits.json"),
        Path("reports/artifacts/day9_object_detection_dataset_analysis.json"),
        Path("reports/artifacts/day9_object_detection_dataset_split.json"),
        Path("reports/artifacts/day9_detection_visual_validation.json"),
        Path("reports/artifacts/day10_opencv_image_analysis.json"),
        Path("reports/artifacts/day10_opencv_visual_validation.json"),
    )

    duplicate_sources = (
        root / "reports/artifacts/day9_object_detection_dataset_analysis.json",
        root / "reports/artifacts/day9_detection_visual_validation.json",
    )

    return {
        "project_root": str(root),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "checks": [check.to_dict() for check in checks],
        "package_versions": package_versions,
        "import_results": import_results,
        "torchvision_detection_api": model_api,
        "disk_free_gib": round(disk.free / (1024**3), 2),
        "torch_checkpoint_cache": {
            "path": str(cache_root) if cache_root else None,
            "files": cache_files,
        },
        "src_detection_tree": detection_tree,
        "existing_detection_dataset_model_code": existing_dataset_model_code,
        "dataset_config_symbols": python_symbols(
            root / "src/detection/dataset_config.py"
        ),
        "annotation_parser_symbols": python_symbols(
            root / "src/detection/annotation_parser.py"
        ),
        "json_schema_previews": {
            relative.as_posix(): json_schema_preview(root / relative)
            for relative in json_paths
        },
        "duplicate_box_records": {
            path.relative_to(root).as_posix(): find_duplicate_box_records(path)
            for path in duplicate_sources
        },
        "requirements_relevant_lines": [
            line.strip()
            for line in read_text(root / "requirements.txt").splitlines()
            if any(
                token in line.lower()
                for token in (
                    "torch",
                    "torchvision",
                    "pillow",
                    "numpy",
                    "opencv",
                )
            )
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 100)
    print("DAY 11 - DETECTION PREREQUISITE INSPECTION")
    print("=" * 100)
    print(f"Project root      : {report['project_root']}")
    print(f"Python            : {report['python_version']}")
    print(f"Python executable : {report['python_executable']}")
    print(f"Disk free         : {report['disk_free_gib']:.2f} GiB")

    print("\n[PATH CHECKS]")
    for check in report["checks"]:
        print(f"[{check['status']:<4}] {check['name']} - {check['detail']}")

    print("\n[PACKAGE VERSIONS]")
    for name, version in report["package_versions"].items():
        print(f"{name:<20}: {version or 'NOT INSTALLED'}")

    print("\n[TORCHVISION DETECTION API]")
    for name, detail in report["torchvision_detection_api"].items():
        print(f"{name}: {detail}")

    print("\n[EXISTING src/detection FILES]")
    for path in report["src_detection_tree"]:
        print(f"- {path}")

    print("\n[EXISTING DETECTION DATASET/MODEL-LIKE FILES]")
    values = report["existing_detection_dataset_model_code"]
    if values:
        for path in values:
            print(f"- {path}")
    else:
        print("- none")

    print("\n[DUPLICATE BOX RECORDS]")
    for source, records in report["duplicate_box_records"].items():
        print(f"- {source}: {len(records)} record(s)")
        for record in records:
            print(json.dumps(record, ensure_ascii=False, indent=2))

    print("\n[JSON SCHEMA PREVIEWS]")
    print(json.dumps(report["json_schema_previews"], ensure_ascii=False, indent=2))

    print("\n[PYTHON SYMBOLS]")
    print("dataset_config.py")
    print(json.dumps(report["dataset_config_symbols"], ensure_ascii=False, indent=2))
    print("annotation_parser.py")
    print(json.dumps(report["annotation_parser_symbols"], ensure_ascii=False, indent=2))

    print("\n[PRETRAINED WEIGHT CACHE]")
    print(json.dumps(report["torch_checkpoint_cache"], ensure_ascii=False, indent=2))

    print("\n[RELEVANT REQUIREMENTS]")
    for line in report["requirements_relevant_lines"]:
        print(line)

    print("\n[COMPACT JSON]")
    compact = {
        "project_root": report["project_root"],
        "python_version": report["python_version"],
        "package_versions": report["package_versions"],
        "torchvision_detection_api": report["torchvision_detection_api"],
        "disk_free_gib": report["disk_free_gib"],
        "existing_detection_dataset_model_code": report[
            "existing_detection_dataset_model_code"
        ],
        "duplicate_box_record_counts": {
            source: len(records)
            for source, records in report["duplicate_box_records"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Manufacturing Vision Defect Analysis System root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project_root.is_dir():
        print(f"[FAIL] Project root does not exist: {args.project_root}")
        return 1
    report = inspect(args.project_root)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

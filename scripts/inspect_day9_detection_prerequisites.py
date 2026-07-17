"""Day 9 구현 전에 실제 저장소·Dependency·디스크 상태를 점검한다.

이 Script는 기존 파일을 수정하지 않는다.
결과 JSON만 reports/artifacts 아래에 생성한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Iterable


DIRECTORIES_TO_INSPECT: tuple[str, ...] = (
    "data",
    "src/data",
    "src/models",
    "src/api",
    "src/dashboard",
    "src/detection",
    "scripts",
    "tests",
    "reports/artifacts",
    "reports/figures",
    "models/checkpoints",
)

FILES_TO_INSPECT: tuple[str, ...] = (
    "requirements.txt",
    "README.md",
    ".gitignore",
)

PACKAGE_NAMES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "Pillow",
    "matplotlib",
    "opencv-python",
    "opencv-contrib-python",
    "lxml",
    "torchmetrics",
    "pycocotools",
    "streamlit",
    "httpx",
    "httpx2",
)


def _safe_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _directory_snapshot(
    directory: Path,
    *,
    project_root: Path,
    max_entries: int = 200,
) -> dict[str, object]:
    if not directory.exists():
        return {"exists": False, "entries": []}

    entries: list[dict[str, object]] = []
    total_files = 0
    total_directories = 0

    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for path in sorted(directory.rglob("*")):
        relative_parts = set(path.relative_to(project_root).parts)
        if relative_parts & ignored_parts:
            continue
        if path.is_dir():
            total_directories += 1
        elif path.is_file():
            total_files += 1
        if len(entries) < max_entries:
            entries.append(
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "type": "directory" if path.is_dir() else "file",
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                }
            )

    return {
        "exists": True,
        "total_files": total_files,
        "total_directories": total_directories,
        "entries_truncated": total_files + total_directories > max_entries,
        "entries": entries,
    }


def _matching_source_files(project_root: Path) -> list[str]:
    keywords = (
        "detect",
        "detection",
        "bounding",
        "bbox",
        "pascal",
        "voc",
        "neu_det",
        "neu-det",
    )
    matches: list[str] = []
    for search_root in (
        project_root / "src",
        project_root / "scripts",
        project_root / "tests",
    ):
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(project_root)
            if set(relative_path.parts) & {
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
            }:
                continue
            relative = relative_path.as_posix().lower()
            if any(keyword in relative for keyword in keywords):
                matches.append(relative_path.as_posix())
    return sorted(set(matches))


def _read_relevant_requirement_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    keywords = (
        "opencv",
        "lxml",
        "torchmetrics",
        "pycocotools",
        "torch",
        "pillow",
        "matplotlib",
        "streamlit",
        "httpx",
    )
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [
        line
        for line in lines
        if any(keyword in line.lower() for keyword in keywords)
    ]


def inspect(project_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project Root가 없습니다: {project_root}")

    disk_usage = shutil.disk_usage(project_root)
    packages = {
        package_name: _safe_package_version(package_name)
        for package_name in PACKAGE_NAMES
    }

    try:
        cv2_module = importlib.import_module("cv2")
        cv2_import = {
            "success": True,
            "version": getattr(cv2_module, "__version__", "unknown"),
        }
    except Exception as exc:  # 설치 충돌도 결과에 남긴다.
        cv2_import = {
            "success": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    stdlib_xml = {
        "xml.etree.ElementTree": True,
        "note": "Pascal VOC XML Parser는 표준 라이브러리로 구현 가능",
    }

    file_status = {}
    for relative in FILES_TO_INSPECT:
        path = project_root / relative
        file_status[relative] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }

    directory_status = {
        relative: _directory_snapshot(
            project_root / relative,
            project_root=project_root,
        )
        for relative in DIRECTORIES_TO_INSPECT
    }

    data_root = project_root / "data" / "raw"
    candidate_dataset_roots = []
    if data_root.exists():
        for path in sorted(data_root.iterdir()):
            if path.is_dir():
                candidate_dataset_roots.append(
                    path.relative_to(project_root).as_posix()
                )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "disk": {
            "total_bytes": disk_usage.total,
            "used_bytes": disk_usage.used,
            "free_bytes": disk_usage.free,
            "free_gib": round(disk_usage.free / (1024 ** 3), 3),
        },
        "packages": packages,
        "cv2_import": cv2_import,
        "xml_support": stdlib_xml,
        "files": file_status,
        "directories": directory_status,
        "raw_dataset_directories": candidate_dataset_roots,
        "existing_detection_related_files": _matching_source_files(project_root),
        "relevant_requirement_lines": _read_relevant_requirement_lines(
            project_root / "requirements.txt"
        ),
    }


def _print_summary(report: dict[str, object], output_path: Path) -> None:
    print("=" * 100)
    print("DAY 9 - OBJECT DETECTION PREREQUISITES")
    print("=" * 100)
    print(f"[PROJECT ROOT] {report['project_root']}")
    environment = report["environment"]
    print(f"[PYTHON] {environment['python_executable']}")
    print(f"[VERSION] {environment['python'].splitlines()[0]}")
    disk = report["disk"]
    print(f"[DISK FREE] {disk['free_gib']} GiB")
    print()

    print("[DEPENDENCIES]")
    for package_name, version in report["packages"].items():
        status = version if version is not None else "NOT INSTALLED"
        print(f"{package_name:<24}: {status}")
    print(f"cv2 import              : {report['cv2_import']}")
    print()

    print("[REPOSITORY TARGETS]")
    for relative, status in report["files"].items():
        print(f"{relative:<28}: exists={status['exists']}")
    for relative, status in report["directories"].items():
        print(
            f"{relative:<28}: exists={status['exists']} "
            f"files={status.get('total_files', 0)} "
            f"dirs={status.get('total_directories', 0)}"
        )
    print()

    print("[RAW DATASET DIRECTORIES]")
    for value in report["raw_dataset_directories"]:
        print(value)
    if not report["raw_dataset_directories"]:
        print("(none)")
    print()

    print("[EXISTING DETECTION-RELATED FILES]")
    for value in report["existing_detection_related_files"]:
        print(value)
    if not report["existing_detection_related_files"]:
        print("(none)")
    print()

    print(f"[REPORT] {output_path}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Manufacturing Vision Defect Analysis System 프로젝트 루트",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 JSON 경로. 생략하면 reports/artifacts에 저장",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else project_root
        / "reports"
        / "artifacts"
        / "day9_detection_prerequisites.json"
    )
    report = inspect(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_summary(report, output_path)


if __name__ == "__main__":
    main()

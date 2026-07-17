"""Day 10 구현 전 저장소·의존성·Day 9 산출물을 빠르게 점검한다."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Any


DAY9_README_START = "<!-- DAY9_OBJECT_DETECTION_DATASET_START -->"
DAY9_README_END = "<!-- DAY9_OBJECT_DETECTION_DATASET_END -->"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _status_for_path(path: Path, *, required: bool) -> CheckResult:
    exists = path.exists()
    status = "PASS" if exists else ("FAIL" if required else "INFO")
    kind = "directory" if path.is_dir() else "file"
    detail = f"{kind} exists" if exists else "not found"
    return CheckResult(str(path), status, detail)


def _read_text_safely(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_check(module_name: str, distribution_name: str | None = None) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - 실제 환경 진단용
        return CheckResult(module_name, "FAIL", f"import failed: {exc!r}")

    version = getattr(module, "__version__", None)
    if version is None and distribution_name:
        version = _distribution_version(distribution_name)
    return CheckResult(module_name, "PASS", f"import ok, version={version or 'unknown'}")


def _search_opencv_references(project_root: Path) -> list[str]:
    """대형 데이터·가상환경을 제외하고 OpenCV 관련 기존 참조를 찾는다."""
    matches: list[str] = []
    roots = [
        project_root / "src",
        project_root / "scripts",
        project_root / "tests",
    ]
    extra_files = [project_root / "requirements.txt", project_root / "pyproject.toml"]

    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.rglob("*.py"))
    candidates.extend(path for path in extra_files if path.is_file())

    for path in sorted(candidates):
        text = _read_text_safely(path).lower()
        if "cv2" in text or "opencv" in text:
            matches.append(str(path.relative_to(project_root)))
    return matches


def inspect(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    checks: list[CheckResult] = []

    checks.append(_status_for_path(root, required=True))
    for relative, required in (
        ("requirements.txt", True),
        ("README.md", True),
        (".gitignore", True),
        ("src/data", True),
        ("src/detection", True),
        ("src/api", False),
        ("src/dashboard", False),
        ("src/opencv_analysis", False),
        ("scripts", True),
        ("tests", True),
        ("reports/artifacts", True),
        ("reports/figures", True),
        ("reports/day9_object_detection_dataset_analysis_summary.md", True),
        ("reports/artifacts/day9_detection_visual_validation.json", True),
    ):
        checks.append(_status_for_path(root / relative, required=required))

    readme_path = root / "README.md"
    readme_text = _read_text_safely(readme_path)
    marker_ok = DAY9_README_START in readme_text and DAY9_README_END in readme_text
    checks.append(
        CheckResult(
            "Day 9 README markers",
            "PASS" if marker_ok else "FAIL",
            "both markers found" if marker_ok else "one or both markers missing",
        )
    )

    requirements_text = _read_text_safely(root / "requirements.txt")
    requirement_lines = [
        line.strip()
        for line in requirements_text.splitlines()
        if "opencv" in line.lower()
    ]
    checks.append(
        CheckResult(
            "OpenCV requirement",
            "PASS" if requirement_lines else "WARN",
            ", ".join(requirement_lines) if requirement_lines else "not listed",
        )
    )

    cv2_check = _module_check("cv2", "opencv-python")
    checks.append(cv2_check)
    checks.append(_module_check("numpy", "numpy"))
    checks.append(_module_check("PIL", "Pillow"))
    checks.append(_module_check("matplotlib", "matplotlib"))

    opencv_distribution = _distribution_version("opencv-python")
    checks.append(
        CheckResult(
            "opencv-python distribution",
            "PASS" if opencv_distribution else "WARN",
            f"version={opencv_distribution}" if opencv_distribution else "not installed",
        )
    )

    disk = shutil.disk_usage(root)
    free_gib = disk.free / (1024**3)
    checks.append(
        CheckResult(
            "Disk free space",
            "PASS" if free_gib >= 2.0 else "WARN",
            f"{free_gib:.2f} GiB free",
        )
    )

    opencv_references = _search_opencv_references(root)
    checks.append(
        CheckResult(
            "Existing OpenCV references",
            "INFO",
            f"{len(opencv_references)} file(s)",
        )
    )

    return {
        "project_root": str(root),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "checks": [item.to_dict() for item in checks],
        "existing_opencv_references": opencv_references,
        "safe_to_add_new_package": not (root / "src/opencv_analysis").exists(),
    }


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 100)
    print("DAY 10 - OPENCV PREREQUISITE INSPECTION")
    print("=" * 100)
    print(f"Project root      : {report['project_root']}")
    print(f"Python            : {report['python']}")
    print(f"Python executable : {report['python_executable']}")
    print()

    for item in report["checks"]:
        print(f"[{item['status']:<4}] {item['name']}")
        print(f"       {item['detail']}")

    print()
    print("[EXISTING OPENCV REFERENCES]")
    references = report["existing_opencv_references"]
    if references:
        for path in references:
            print(f"- {path}")
    else:
        print("- none")

    print()
    print(f"Safe to add src/opencv_analysis : {report['safe_to_add_new_package']}")
    print()
    print("[JSON]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="프로젝트 루트 경로",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project_root.exists():
        print(f"[FAIL] Project root does not exist: {args.project_root}")
        return 1

    report = inspect(args.project_root)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Day 8 구현 전 현재 저장소와 Local Dependency 상태를 빠르게 점검한다."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECK_PATHS = (
    "requirements.txt",
    "src/dashboard/__init__.py",
    "src/api/config.py",
    "src/api/schemas.py",
    "src/api/app.py",
    "reports/artifacts/day7_fastapi_inference_validation.json",
)


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _count_streamlit_test_references(tests_root: Path) -> int:
    if not tests_root.is_dir():
        return 0
    count = 0
    for path in tests_root.rglob("test_*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "streamlit" in text.lower():
            count += 1
    return count


def main() -> None:
    print("=" * 88)
    print("DAY 8 - STREAMLIT DASHBOARD PREREQUISITE INSPECTION")
    print("=" * 88)

    print("\n[FILES]")
    for relative_path in CHECK_PATHS:
        path = PROJECT_ROOT / relative_path
        status = "PASS" if path.exists() else "MISSING"
        print(f"[{status}] {relative_path}")

    dashboard_root = PROJECT_ROOT / "src" / "dashboard"
    dashboard_files = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in dashboard_root.glob("*.py")
    ) if dashboard_root.is_dir() else []

    print("\n[DASHBOARD FILES]")
    if dashboard_files:
        for path in dashboard_files:
            print(f"- {path}")
    else:
        print("- src/dashboard does not exist or has no Python files")

    print("\n[DEPENDENCIES]")
    for package_name in ("streamlit", "httpx", "Pillow"):
        print(f"{package_name:<12}: {_package_version(package_name)}")

    print("\n[TEST REFERENCES]")
    reference_count = _count_streamlit_test_references(PROJECT_ROOT / "tests")
    print(f"Tests containing 'streamlit': {reference_count}")

    print("\n[PROJECT ROOT]")
    print(PROJECT_ROOT)


if __name__ == "__main__":
    main()

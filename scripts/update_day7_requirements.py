"""Day 7 API Dependency를 requirements.txt에 중복 없이 추가한다."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

REQUIRED_PACKAGES = (
    "fastapi",
    "uvicorn",
    "python-multipart",
    "httpx",
    "pydantic",
    "Pillow",
)


def normalize_package_name(value: str) -> str:
    """Version 조건과 Extra를 제외한 Package 이름을 정규화한다."""

    candidate = value.strip()
    if not candidate or candidate.startswith("#"):
        return ""

    candidate = candidate.split("#", maxsplit=1)[0].strip()
    candidate = candidate.split("[", maxsplit=1)[0]
    candidate = re.split(r"[<>=!~; ]", candidate, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", candidate).lower()


def main() -> None:
    if not REQUIREMENTS_PATH.is_file():
        raise FileNotFoundError(
            f"requirements.txt does not exist: {REQUIREMENTS_PATH}"
        )

    original_text = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    original_lines = original_text.splitlines()

    existing_names = {
        normalize_package_name(line)
        for line in original_lines
        if normalize_package_name(line)
    }

    missing_packages = [
        package
        for package in REQUIRED_PACKAGES
        if normalize_package_name(package) not in existing_names
    ]

    if not missing_packages:
        print("[PASS] Day 7 API dependencies already exist in requirements.txt")
        return

    updated_lines = list(original_lines)
    if updated_lines and updated_lines[-1].strip():
        updated_lines.append("")

    updated_lines.append("# Day 7 - FastAPI Image Inference API")
    updated_lines.extend(missing_packages)

    REQUIREMENTS_PATH.write_text(
        "\n".join(updated_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print("[PASS] requirements.txt updated")
    for package in missing_packages:
        print(f"[ADDED] {package}")


if __name__ == "__main__":
    main()

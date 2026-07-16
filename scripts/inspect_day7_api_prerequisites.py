"""Day 7 실제 모델 연결에 필요한 기존 코드와 Dependency를 점검한다.

실행:
    python -m scripts.inspect_day7_api_prerequisites
"""

from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_NAMES = (
    "fastapi",
    "uvicorn",
    "python-multipart",
    "httpx",
    "pydantic",
    "Pillow",
    "torch",
    "torchvision",
)

TARGET_FILES = (
    Path("requirements.txt"),
    Path("pyproject.toml"),
    Path("src/models/resnet18_transfer.py"),
    Path("src/data/image_transforms.py"),
    Path("scripts/run_day4_resnet18_evaluation.py"),
)

KEYWORDS = (
    "checkpoint",
    "restore",
    "load_state_dict",
    "resnet18",
    "transform",
    "test_transform",
    "testclient",
    "httpx",
    "fastapi",
)


def _print_heading(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def _print_python_symbols(path: Path) -> None:
    text = _read_text(path)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        print(f"[PARSE ERROR] {path}: {exc}")
        return

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            print(
                f"- {type(node).__name__}: {node.name} "
                f"(lines {node.lineno}-{getattr(node, 'end_lineno', node.lineno)})"
            )


def _print_keyword_hits(path: Path) -> None:
    text = _read_text(path)
    lines = text.splitlines()

    for line_number, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(keyword in lowered for keyword in KEYWORDS):
            print(f"{path}:{line_number}: {line.rstrip()}")


def main() -> None:
    _print_heading("DAY 7 DEPENDENCY VERSIONS")
    for package_name in PACKAGE_NAMES:
        print(f"{package_name:<20}: {_package_version(package_name)}")

    _print_heading("TARGET FILE EXISTENCE")
    for relative_path in TARGET_FILES:
        absolute_path = PROJECT_ROOT / relative_path
        print(
            f"{relative_path!s:<55}: "
            f"{'FOUND' if absolute_path.exists() else 'NOT FOUND'}"
        )

    _print_heading("CURRENT SRC/API TREE")
    api_root = PROJECT_ROOT / "src" / "api"
    if not api_root.exists():
        print("src/api does not exist")
    else:
        for path in sorted(api_root.rglob("*")):
            if path.is_file():
                print(path.relative_to(PROJECT_ROOT))

    _print_heading("RELEVANT PYTHON SYMBOLS")
    for relative_path in TARGET_FILES:
        absolute_path = PROJECT_ROOT / relative_path
        if absolute_path.suffix == ".py" and absolute_path.exists():
            print()
            print(f"[{relative_path}]")
            _print_python_symbols(absolute_path)

    _print_heading("CHECKPOINT / TRANSFORM / API KEYWORD HITS")
    search_roots = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
    )

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in sorted(search_root.rglob("*.py")):
            _print_keyword_hits(path)

    _print_heading("REQUIREMENTS CONTENT")
    requirements_path = PROJECT_ROOT / "requirements.txt"
    if requirements_path.exists():
        print(_read_text(requirements_path))
    else:
        print("requirements.txt not found")

    _print_heading("IMPORTANT FILE CONTENT")
    for relative_path in (
        Path("src/models/resnet18_transfer.py"),
        Path("src/data/image_transforms.py"),
        Path("scripts/run_day4_resnet18_evaluation.py"),
    ):
        absolute_path = PROJECT_ROOT / relative_path
        print()
        print(f"--- {relative_path} ---")
        if absolute_path.exists():
            print(_read_text(absolute_path))
        else:
            print("NOT FOUND")


if __name__ == "__main__":
    main()

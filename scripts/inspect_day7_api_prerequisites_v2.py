"""Day 7 실제 모델 연결에 필요한 핵심 정보만 간결하게 출력한다.

기존 1차 검사 Script처럼 긴 파일 전체를 출력하지 않는다.

실행:
    python -m scripts.inspect_day7_api_prerequisites_v2
"""

from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path
from typing import Iterable


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

PRIMARY_FILES = (
    Path("requirements.txt"),
    Path("pyproject.toml"),
    Path("src/models/resnet18_transfer.py"),
    Path("src/data/image_transforms.py"),
    Path("scripts/run_day4_resnet18_evaluation.py"),
)

SEARCH_ROOTS = (
    Path("src"),
    Path("scripts"),
    Path("tests"),
)

RELEVANT_REQUIREMENT_NAMES = (
    "fastapi",
    "uvicorn",
    "multipart",
    "httpx",
    "pydantic",
    "pillow",
    "torch",
    "torchvision",
)

RELEVANT_SYMBOL_TERMS = (
    "checkpoint",
    "restore",
    "transform",
    "resnet18",
    "test",
)

RELEVANT_CONSTANT_TERMS = (
    "checkpoint",
    "threshold",
    "image_size",
    "model",
)


def print_heading(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def iter_python_files() -> Iterable[Path]:
    for relative_root in SEARCH_ROOTS:
        root = PROJECT_ROOT / relative_root
        if not root.exists():
            continue

        yield from sorted(root.rglob("*.py"))


def parse_python(path: Path) -> tuple[str, ast.Module] | None:
    try:
        text = read_text(path)
        return text, ast.parse(text)
    except (OSError, SyntaxError) as exc:
        print(f"[PARSE ERROR] {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


def dotted_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr

    return ""


def format_import(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        names = []
        for alias in node.names:
            item = alias.name
            if alias.asname:
                item += f" as {alias.asname}"
            names.append(item)
        return f"import {', '.join(names)}"

    module = "." * node.level + (node.module or "")
    names = []
    for alias in node.names:
        item = alias.name
        if alias.asname:
            item += f" as {alias.asname}"
        names.append(item)
    return f"from {module} import {', '.join(names)}"


def print_imports(path: Path) -> None:
    parsed = parse_python(path)
    if parsed is None:
        return

    _, tree = parsed
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = format_import(node)
            lowered = rendered.lower()

            if any(
                term in lowered
                for term in (
                    "resnet",
                    "transform",
                    "checkpoint",
                    "training",
                    "data",
                    "torch",
                    "api",
                )
            ):
                print(rendered)


def print_symbols(path: Path) -> None:
    parsed = parse_python(path)
    if parsed is None:
        return

    _, tree = parsed

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_name = node.name
            lowered = symbol_name.lower()

            if (
                path.name in {"resnet18_transfer.py", "image_transforms.py"}
                or any(term in lowered for term in RELEVANT_SYMBOL_TERMS)
            ):
                print(
                    f"{type(node).__name__:<16} "
                    f"{symbol_name:<45} "
                    f"lines={node.lineno}-{getattr(node, 'end_lineno', node.lineno)}"
                )


def print_relevant_assignments(path: Path) -> None:
    parsed = parse_python(path)
    if parsed is None:
        return

    text, tree = parsed

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        target_names: list[str] = []

        if isinstance(node, ast.Assign):
            for target in node.targets:
                target_name = dotted_name(target)
                if target_name:
                    target_names.append(target_name)
        else:
            target_name = dotted_name(node.target)
            if target_name:
                target_names.append(target_name)

        if not target_names:
            continue

        joined_names = " ".join(target_names).lower()
        if not any(term in joined_names for term in RELEVANT_CONSTANT_TERMS):
            continue

        source = ast.get_source_segment(text, node)
        if source:
            compact = " ".join(line.strip() for line in source.splitlines())
            if len(compact) > 300:
                compact = compact[:297] + "..."
            print(compact)


def print_selected_function_sources() -> None:
    print_heading("SELECTED CHECKPOINT / TRANSFORM FUNCTION DEFINITIONS")

    matches: list[tuple[Path, str, int, str]] = []

    for path in iter_python_files():
        parsed = parse_python(path)
        if parsed is None:
            continue

        text, tree = parsed

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            lowered = node.name.lower()
            is_checkpoint_function = (
                "checkpoint" in lowered
                and any(term in lowered for term in ("restore", "load", "read"))
            )
            is_transform_function = (
                "transform" in lowered
                and any(term in lowered for term in ("test", "eval", "create", "build"))
            )

            if not (is_checkpoint_function or is_transform_function):
                continue

            source = ast.get_source_segment(text, node)
            if not source:
                continue

            # 함수 하나가 비정상적으로 길어도 Terminal 전체를 덮지 않게 제한한다.
            source_lines = source.splitlines()
            if len(source_lines) > 120:
                source = "\n".join(source_lines[:120])
                source += "\n# ... OUTPUT TRUNCATED AFTER 120 LINES ..."

            matches.append(
                (
                    path.relative_to(PROJECT_ROOT),
                    node.name,
                    node.lineno,
                    source,
                )
            )

    if not matches:
        print("No matching checkpoint/transform function definition found.")
        return

    for relative_path, function_name, line_number, source in matches:
        print()
        print(f"--- {relative_path}:{line_number} [{function_name}] ---")
        print(source)


def print_test_client_usage() -> None:
    print_heading("EXISTING FASTAPI / TESTCLIENT / HTTPX USAGE")

    hit_count = 0

    for path in iter_python_files():
        relative_path = path.relative_to(PROJECT_ROOT)

        # 새 Day 7 후보 폴더가 Project 아래 있다면 기존 구조로 오인하지 않는다.
        if "day7_phase1_candidate" in relative_path.parts:
            continue

        text = read_text(path)
        lines = text.splitlines()

        relevant_lines: list[tuple[int, str]] = []
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(
                term in lowered
                for term in (
                    "from fastapi",
                    "import fastapi",
                    "testclient",
                    "httpx.client",
                    "httpx.asyncclient",
                )
            ):
                relevant_lines.append((line_number, line.rstrip()))

        if relevant_lines:
            hit_count += 1
            print()
            print(f"[{relative_path}]")
            for line_number, line in relevant_lines[:30]:
                print(f"{line_number}: {line}")

    if hit_count == 0:
        print("No existing FastAPI/TestClient/httpx usage found.")


def main() -> None:
    print_heading("DAY 7 DEPENDENCY VERSIONS")
    for package_name in PACKAGE_NAMES:
        print(f"{package_name:<20}: {package_version(package_name)}")

    print_heading("PRIMARY FILE EXISTENCE")
    for relative_path in PRIMARY_FILES:
        absolute_path = PROJECT_ROOT / relative_path
        print(
            f"{str(relative_path):<55}: "
            f"{'FOUND' if absolute_path.exists() else 'NOT FOUND'}"
        )

    print_heading("RELEVANT REQUIREMENTS")
    requirements_path = PROJECT_ROOT / "requirements.txt"

    if requirements_path.exists():
        matching_lines = [
            line
            for line in read_text(requirements_path).splitlines()
            if any(term in line.lower() for term in RELEVANT_REQUIREMENT_NAMES)
        ]

        if matching_lines:
            for line in matching_lines:
                print(line)
        else:
            print("No relevant requirement line found.")
    else:
        print("requirements.txt not found")

    print_heading("CURRENT SRC/API TREE")
    api_root = PROJECT_ROOT / "src" / "api"

    if not api_root.exists():
        print("src/api does not exist")
    else:
        for path in sorted(api_root.rglob("*")):
            if path.is_file():
                print(path.relative_to(PROJECT_ROOT))

    print_heading("PRIMARY FILE SYMBOLS")
    for relative_path in (
        Path("src/models/resnet18_transfer.py"),
        Path("src/data/image_transforms.py"),
        Path("scripts/run_day4_resnet18_evaluation.py"),
    ):
        absolute_path = PROJECT_ROOT / relative_path
        print()
        print(f"[{relative_path}]")

        if not absolute_path.exists():
            print("NOT FOUND")
            continue

        print_symbols(absolute_path)

    print_heading("DAY 4 RELEVANT IMPORTS")
    day4_path = PROJECT_ROOT / "scripts" / "run_day4_resnet18_evaluation.py"

    if day4_path.exists():
        print_imports(day4_path)
    else:
        print("scripts/run_day4_resnet18_evaluation.py not found")

    print_heading("RELEVANT CONSTANT ASSIGNMENTS")
    for relative_path in (
        Path("src/models/resnet18_transfer.py"),
        Path("src/data/image_transforms.py"),
        Path("scripts/run_day4_resnet18_evaluation.py"),
    ):
        absolute_path = PROJECT_ROOT / relative_path
        print()
        print(f"[{relative_path}]")

        if absolute_path.exists():
            print_relevant_assignments(absolute_path)
        else:
            print("NOT FOUND")

    print_selected_function_sources()
    print_test_client_usage()

    print()
    print("=" * 100)
    print("[PASS] Day 7 concise prerequisite inspection completed")
    print("=" * 100)


if __name__ == "__main__":
    main()

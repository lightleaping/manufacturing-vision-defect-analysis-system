from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_MODULE = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "detection_page.py"
)
ENTRYPOINT = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "pages"
    / "2_Detection.py"
)


def imports_from_source(path: Path) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8")
    )
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def test_detection_page_files_exist() -> None:
    assert PAGE_MODULE.is_file()
    assert ENTRYPOINT.is_file()


def test_detection_page_uses_api_client_only() -> None:
    imports = imports_from_source(PAGE_MODULE)
    forbidden_prefixes = (
        "torch",
        "torchvision",
        "src.api.detection",
        "src.detection",
    )

    assert not any(
        module.startswith(forbidden_prefixes)
        for module in imports
    )
    assert (
        "src.dashboard.detection_api_client"
        in imports
    )


def test_detection_page_has_required_ui_contract() -> None:
    source = PAGE_MODULE.read_text(
        encoding="utf-8"
    )

    assert 'value=0.50' in source
    assert '"Detection 실행"' in source
    assert "Prediction Table" in source
    assert "render_detection_overlay" in source
    assert "OpenCV" in source
    assert "Contour 후보" in source
    assert "Ground Truth" in source
    assert "Checkpoint" not in imports_from_source(PAGE_MODULE)


def test_entrypoint_uses_detection_render_function() -> None:
    source = ENTRYPOINT.read_text(
        encoding="utf-8"
    )

    assert "render_detection_page" in source
    assert "load_dashboard_settings" in source

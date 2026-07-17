"""Tests for rebuilding the Day 13 API core-context artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rebuild_day13_api_core_context import (
    ContextRebuildError,
    build_context_text,
    extract_source_header_values,
    rebuild_context_artifact,
    resolve_source_path,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _damaged_context(*paths: str) -> str:
    sections = []
    for path in paths:
        sections.append(
            "=" * 100
            + "\n"
            + f"FILE: {path}\n"
            + "=" * 100
            + "\n"
            + "?쒖 손상된 내용\n"
        )
    return "\n".join(sections)


def _build_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    _write(
        project / "src/api/app.py",
        '"""정상 한글 설명."""\n\nAPP_NAME = "vision"\n',
    )
    _write(
        project / "src/api/schemas.py",
        '"""응답 스키마."""\n\nSTATUS = "ok"\n',
    )
    _write(
        project / "reports/artifacts/day13_api_core_context.txt",
        _damaged_context(
            r".\src\api\app.py",
            r".\src\api\schemas.py",
        ),
    )
    return project


def test_extract_source_headers_preserves_order_and_removes_duplicates() -> None:
    text = _damaged_context(
        r".\src\api\app.py",
        r".\src\api\schemas.py",
        r".\src\api\app.py",
    )

    assert extract_source_header_values(text) == [
        r".\src\api\app.py",
        r".\src\api\schemas.py",
    ]


def test_extract_source_headers_requires_at_least_one_header() -> None:
    with pytest.raises(ContextRebuildError, match="FILE Header"):
        extract_source_header_values("no headers")


def test_resolve_source_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ContextRebuildError, match="상위 디렉터리"):
        resolve_source_path(
            project_root=tmp_path,
            header_value=r"..\outside.py",
        )


def test_resolve_source_path_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ContextRebuildError, match="절대 경로"):
        resolve_source_path(
            project_root=tmp_path,
            header_value=r"C:\outside.py",
        )


def test_build_context_uses_current_utf8_sources(tmp_path: Path) -> None:
    project = _build_project(tmp_path)

    rebuilt, records = build_context_text(
        project_root=project,
        header_values=[
            r".\src\api\app.py",
            r".\src\api\schemas.py",
        ],
    )

    assert "정상 한글 설명" in rebuilt
    assert "응답 스키마" in rebuilt
    assert "?쒖 손상된 내용" not in rebuilt
    assert [record["path"] for record in records] == [
        "src/api/app.py",
        "src/api/schemas.py",
    ]


def test_build_context_rejects_mojibake_in_current_source(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    _write(project / "src/api/app.py", "?쒖 손상 Source\n")

    with pytest.raises(ContextRebuildError, match="문자 깨짐 토큰"):
        build_context_text(
            project_root=project,
            header_values=[r".\src\api\app.py"],
        )


def test_dry_run_does_not_modify_target_or_create_summary(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    target = project / "reports/artifacts/day13_api_core_context.txt"
    before = target.read_bytes()

    payload = rebuild_context_artifact(
        project_root=project,
        target_path=Path(
            "reports/artifacts/day13_api_core_context.txt"
        ),
        summary_path=Path(
            "reports/artifacts/day14_day13_api_core_context_rebuild.json"
        ),
        dry_run=True,
    )

    assert target.read_bytes() == before
    assert payload["status"] == "DRY_RUN_PASS"
    assert payload["target_modified"] is False
    assert not (
        project
        / "reports/artifacts/day14_day13_api_core_context_rebuild.json"
    ).exists()


def test_rebuild_creates_exact_backup_and_clean_target(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)
    target = project / "reports/artifacts/day13_api_core_context.txt"
    original = target.read_bytes()

    payload = rebuild_context_artifact(
        project_root=project,
        target_path=Path(
            "reports/artifacts/day13_api_core_context.txt"
        ),
        summary_path=Path(
            "reports/artifacts/day14_day13_api_core_context_rebuild.json"
        ),
    )

    backup = project / payload["backup_path"]
    assert backup.read_bytes() == original

    rebuilt = target.read_text(encoding="utf-8")
    assert "정상 한글 설명" in rebuilt
    assert "응답 스키마" in rebuilt
    assert "?쒖 손상된 내용" not in rebuilt
    assert payload["rebuilt"]["mojibake_tokens"] == []


def test_rebuild_writes_valid_summary_json(tmp_path: Path) -> None:
    project = _build_project(tmp_path)

    payload = rebuild_context_artifact(
        project_root=project,
        target_path=Path(
            "reports/artifacts/day13_api_core_context.txt"
        ),
        summary_path=Path(
            "reports/artifacts/day14_day13_api_core_context_rebuild.json"
        ),
    )

    summary_path = project / payload["summary_path"]
    saved = json.loads(summary_path.read_text(encoding="utf-8"))

    assert saved["status"] == "PASS"
    assert saved["source_file_count"] == 2
    assert saved["source_files_modified"] is False
    assert saved["target_modified"] is True


def test_rebuilt_target_has_no_utf8_bom(tmp_path: Path) -> None:
    project = _build_project(tmp_path)

    payload = rebuild_context_artifact(
        project_root=project,
        target_path=Path(
            "reports/artifacts/day13_api_core_context.txt"
        ),
        summary_path=Path(
            "reports/artifacts/day14_day13_api_core_context_rebuild.json"
        ),
    )

    target = project / payload["target_path"]
    assert not target.read_bytes().startswith(b"\xef\xbb\xbf")


def test_missing_source_aborts_before_target_write(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    target = project / "reports/artifacts/day13_api_core_context.txt"
    _write(
        target,
        _damaged_context(r".\src\api\missing.py"),
    )
    before = target.read_bytes()

    with pytest.raises(ContextRebuildError, match="Source 파일이 없습니다"):
        rebuild_context_artifact(
            project_root=project,
            target_path=Path(
                "reports/artifacts/day13_api_core_context.txt"
            ),
            summary_path=Path(
                "reports/artifacts/day14_day13_api_core_context_rebuild.json"
            ),
        )

    assert target.read_bytes() == before


def test_header_format_is_preserved_as_windows_relative_path(
    tmp_path: Path,
) -> None:
    project = _build_project(tmp_path)

    rebuilt, _records = build_context_text(
        project_root=project,
        header_values=[r".\src\api\app.py"],
    )

    assert "FILE: .\\src\\api\\app.py" in rebuilt

"""Rebuild the damaged Day 13 API core-context text artifact from current sources.

The existing artifact contains ``FILE:`` headers followed by source snapshots.
This script uses only those headers to recover the original ordered source-file
list, then reads the current source files as UTF-8 and rebuilds the artifact.

Existing application source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence


DEFAULT_TARGET = Path("reports/artifacts/day13_api_core_context.txt")
DEFAULT_SUMMARY = Path(
    "reports/artifacts/day14_day13_api_core_context_rebuild.json"
)
SEPARATOR = "=" * 100
FILE_HEADER_PATTERN = re.compile(
    r"^FILE:\s*(?P<path>.+?)\s*$",
    re.MULTILINE,
)

# Inspector가 원시 진단 Artifact에서 사용한 대표적인 문자 깨짐 신호다.
# 정상 원본 Source에서 이 토큰이 나오면 덮어쓰지 않고 중단한다.
MOJIBAKE_TOKENS = (
    "\ufffd",
    "媛",
    "吏",
    "쒖",
    "湲",
    "怨",
    "쨌",
)


class ContextRebuildError(RuntimeError):
    """Artifact를 안전하게 재생성할 수 없을 때 발생한다."""


def _sha256_bytes(data: bytes) -> str:
    """Byte 데이터의 SHA-256 값을 반환한다."""

    return hashlib.sha256(data).hexdigest()


def _read_utf8_text(path: Path) -> str:
    """UTF-8 또는 UTF-8 BOM 텍스트를 엄격하게 읽는다."""

    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ContextRebuildError(
            f"UTF-8로 읽을 수 없는 파일입니다: {path}"
        ) from exc
    except OSError as exc:
        raise ContextRebuildError(
            f"파일을 읽을 수 없습니다: {path}"
        ) from exc


def extract_source_header_values(context_text: str) -> list[str]:
    """손상 Artifact의 FILE Header에서 경로 문자열을 순서대로 추출한다."""

    values: list[str] = []
    seen: set[str] = set()

    for match in FILE_HEADER_PATTERN.finditer(context_text):
        raw_value = match.group("path").strip().strip('"').strip("'")
        if not raw_value:
            continue

        comparison_key = raw_value.casefold()
        if comparison_key in seen:
            continue

        seen.add(comparison_key)
        values.append(raw_value)

    if not values:
        raise ContextRebuildError(
            "기존 Artifact에서 FILE Header를 찾지 못했습니다."
        )

    return values


def resolve_source_path(
    *,
    project_root: Path,
    header_value: str,
) -> tuple[Path, str]:
    """Windows 형식 Header 경로를 안전한 프로젝트 상대 경로로 변환한다."""

    windows_path = PureWindowsPath(header_value)

    if windows_path.drive or windows_path.root:
        raise ContextRebuildError(
            f"절대 경로는 허용하지 않습니다: {header_value}"
        )

    clean_parts: list[str] = []
    for part in windows_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ContextRebuildError(
                f"상위 디렉터리 이동은 허용하지 않습니다: {header_value}"
            )
        clean_parts.append(part)

    if not clean_parts:
        raise ContextRebuildError(
            f"비어 있는 Source 경로입니다: {header_value}"
        )

    root = project_root.resolve()
    source_path = root.joinpath(*clean_parts).resolve()

    try:
        relative_path = source_path.relative_to(root)
    except ValueError as exc:
        raise ContextRebuildError(
            f"프로젝트 밖의 경로는 허용하지 않습니다: {header_value}"
        ) from exc

    if not source_path.is_file():
        raise ContextRebuildError(
            f"Header가 가리키는 Source 파일이 없습니다: {relative_path.as_posix()}"
        )

    header_path = ".\\" + "\\".join(relative_path.parts)
    return source_path, header_path


def find_mojibake_tokens(text: str) -> list[str]:
    """재생성 후보 텍스트에 남은 대표 문자 깨짐 토큰을 반환한다."""

    return sorted(
        token
        for token in MOJIBAKE_TOKENS
        if token in text
    )


def build_context_text(
    *,
    project_root: Path,
    header_values: Sequence[str],
) -> tuple[str, list[dict[str, Any]]]:
    """현재 정상 Source를 읽어 Context Artifact 본문을 만든다."""

    sections: list[str] = []
    source_records: list[dict[str, Any]] = []

    for header_value in header_values:
        source_path, normalized_header = resolve_source_path(
            project_root=project_root,
            header_value=header_value,
        )
        source_text = _read_utf8_text(source_path)
        normalized_text = source_text.replace("\r\n", "\n").replace("\r", "\n")
        normalized_text = normalized_text.rstrip("\n") + "\n"

        token_hits = find_mojibake_tokens(normalized_text)
        if token_hits:
            relative = source_path.resolve().relative_to(
                project_root.resolve()
            )
            raise ContextRebuildError(
                "정상 Source 후보에서 문자 깨짐 토큰이 발견됐습니다. "
                f"파일={relative.as_posix()}, 토큰={token_hits}"
            )

        sections.append(
            f"{SEPARATOR}\n"
            f"FILE: {normalized_header}\n"
            f"{SEPARATOR}\n"
            f"{normalized_text}"
        )
        source_records.append(
            {
                "path": source_path.resolve()
                .relative_to(project_root.resolve())
                .as_posix(),
                "header": normalized_header,
                "size_bytes": source_path.stat().st_size,
                "sha256": _sha256_bytes(source_path.read_bytes()),
            }
        )

    rebuilt_text = "\n".join(section.rstrip("\n") for section in sections) + "\n"
    remaining_hits = find_mojibake_tokens(rebuilt_text)
    if remaining_hits:
        raise ContextRebuildError(
            "재생성 결과에 문자 깨짐 토큰이 남아 있습니다: "
            f"{remaining_hits}"
        )

    return rebuilt_text, source_records


def _write_bytes_atomically(path: Path, data: bytes) -> None:
    """같은 디렉터리의 임시 파일에 쓴 뒤 최종 파일로 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """UTF-8 JSON을 원자적으로 저장한다."""

    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _write_bytes_atomically(path, encoded)


def rebuild_context_artifact(
    *,
    project_root: Path,
    target_path: Path,
    summary_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """기존 Header 목록을 기준으로 Context Artifact를 재생성한다."""

    root = project_root.resolve()
    target = (
        target_path
        if target_path.is_absolute()
        else root / target_path
    ).resolve()
    summary = (
        summary_path
        if summary_path.is_absolute()
        else root / summary_path
    ).resolve()

    if not target.is_file():
        raise ContextRebuildError(
            f"재생성 대상 Artifact가 없습니다: {target}"
        )

    original_bytes = target.read_bytes()
    original_text = _read_utf8_text(target)
    original_hash = _sha256_bytes(original_bytes)
    original_hits = find_mojibake_tokens(original_text)
    header_values = extract_source_header_values(original_text)

    rebuilt_text, source_records = build_context_text(
        project_root=root,
        header_values=header_values,
    )
    rebuilt_bytes = rebuilt_text.encode("utf-8")
    rebuilt_hash = _sha256_bytes(rebuilt_bytes)

    backup_path = (
        target.parent
        / "backups"
        / (
            f"{target.stem}.before_utf8_rebuild."
            f"{original_hash[:12]}{target.suffix}"
        )
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "target_path": target.relative_to(root).as_posix(),
        "summary_path": summary.relative_to(root).as_posix(),
        "dry_run": dry_run,
        "status": "DRY_RUN_PASS" if dry_run else "PASS",
        "strategy": (
            "Extract ordered FILE headers from the damaged artifact and "
            "rebuild it from current UTF-8 source files."
        ),
        "source_file_count": len(source_records),
        "sources": source_records,
        "original": {
            "size_bytes": len(original_bytes),
            "sha256": original_hash,
            "mojibake_tokens": original_hits,
        },
        "rebuilt": {
            "size_bytes": len(rebuilt_bytes),
            "sha256": rebuilt_hash,
            "mojibake_tokens": find_mojibake_tokens(rebuilt_text),
            "utf8_bom": rebuilt_bytes.startswith(b"\xef\xbb\xbf"),
        },
        "backup_path": backup_path.relative_to(root).as_posix(),
        "target_modified": not dry_run,
        "source_files_modified": False,
    }

    if dry_run:
        return payload

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        if backup_path.read_bytes() != original_bytes:
            raise ContextRebuildError(
                f"동일 이름의 Backup 내용이 다릅니다: {backup_path}"
            )
    else:
        _write_bytes_atomically(backup_path, original_bytes)

    _write_bytes_atomically(target, rebuilt_bytes)

    verification_bytes = target.read_bytes()
    if verification_bytes != rebuilt_bytes:
        raise ContextRebuildError(
            "재생성 후 Artifact Byte 검증에 실패했습니다."
        )

    verification_text = _read_utf8_text(target)
    verification_hits = find_mojibake_tokens(verification_text)
    if verification_hits:
        raise ContextRebuildError(
            "재생성 후 Artifact에서 문자 깨짐 토큰이 발견됐습니다: "
            f"{verification_hits}"
        )

    _write_json_atomically(summary, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자를 정의한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Backup과 파일 교체 없이 재생성 가능 여부만 검증한다.",
    )
    return parser


def print_summary(payload: dict[str, Any]) -> None:
    """PowerShell에서 확인할 핵심 결과를 출력한다."""

    print("=" * 100)
    print("DAY 14 - REBUILD DAY 13 API CORE CONTEXT")
    print("=" * 100)
    print(f"Status                    : {payload['status']}")
    print(f"Target                    : {payload['target_path']}")
    print(f"Source files              : {payload['source_file_count']}")
    print(
        "Original mojibake tokens : "
        f"{payload['original']['mojibake_tokens']}"
    )
    print(
        "Rebuilt mojibake tokens  : "
        f"{payload['rebuilt']['mojibake_tokens']}"
    )
    print(f"Original SHA-256          : {payload['original']['sha256']}")
    print(f"Rebuilt SHA-256           : {payload['rebuilt']['sha256']}")
    print(f"Backup                    : {payload['backup_path']}")
    print(f"Target modified           : {payload['target_modified']}")
    print(f"Source modified           : {payload['source_files_modified']}")
    if not payload["dry_run"]:
        print(f"[ARTIFACT] {payload['target_path']}")
        print(f"[SUMMARY] {payload['summary_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 진입점."""

    args = build_parser().parse_args(argv)

    try:
        payload = rebuild_context_artifact(
            project_root=args.project_root,
            target_path=args.target,
            summary_path=args.summary,
            dry_run=args.dry_run,
        )
    except ContextRebuildError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

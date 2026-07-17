"""Day 10 실제 OpenCV 분석에 사용할 이미지 후보를 안전하게 점검한다.

이 Script는 이미지를 수정하거나 분석 Artifact를 생성하지 않는다.
프로젝트의 data/raw 아래에서 Casting NORMAL·DEFECT와 NEU-DET 이미지 후보를
찾고, 실제 파일 경로·크기·Pillow Mode를 출력한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

NORMAL_DIRECTORY_NAMES = {
    "ok_front",
    "normal",
    "good",
    "ok",
}

DEFECT_DIRECTORY_NAMES = {
    "def_front",
    "defect",
    "defective",
    "bad",
    "ng",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Day 10 real-image sample candidates."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Manufacturing Vision Defect Analysis System project root.",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=10,
        help="Maximum number of candidate paths to display per category.",
    )
    return parser.parse_args()


def _iter_image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []

    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path).lower(),
    )


def _has_named_component(path: Path, names: set[str]) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return bool(lowered_parts & names)


def _inspect_image(path: Path, project_root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "relative_path": path.relative_to(project_root).as_posix(),
        "absolute_path": str(path.resolve()),
        "suffix": path.suffix.lower(),
        "size_bytes": int(path.stat().st_size),
    }

    try:
        with Image.open(path) as image:
            image.load()
            item.update(
                {
                    "width": int(image.width),
                    "height": int(image.height),
                    "mode": str(image.mode),
                    "format": str(image.format or ""),
                    "pillow_decode": True,
                }
            )
    except (OSError, UnidentifiedImageError) as error:
        item.update(
            {
                "width": None,
                "height": None,
                "mode": None,
                "format": None,
                "pillow_decode": False,
                "decode_error": repr(error),
            }
        )

    return item


def _find_text_references(project_root: Path) -> list[dict[str, Any]]:
    """기존 코드에서 Casting 디렉터리 이름을 사용하는 위치를 찾는다."""
    search_roots = (
        project_root / "src" / "data",
        project_root / "scripts",
        project_root / "tests",
    )
    keywords = ("ok_front", "def_front", "casting_data")
    references: list[dict[str, Any]] = []

    for root in search_roots:
        if not root.is_dir():
            continue

        for path in sorted(root.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8-sig", errors="replace")

            for line_number, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                matched = [keyword for keyword in keywords if keyword in lowered]
                if matched:
                    references.append(
                        {
                            "path": path.relative_to(project_root).as_posix(),
                            "line_number": line_number,
                            "keywords": matched,
                            "line": line.strip(),
                        }
                    )

    return references


def _print_category(
    title: str,
    candidates: list[dict[str, Any]],
    *,
    max_display: int,
) -> None:
    print(f"\n[{title}]")
    print(f"Candidate count : {len(candidates):,}")

    if not candidates:
        print("Recommended     : NONE")
        return

    valid = [candidate for candidate in candidates if candidate["pillow_decode"]]
    recommended = valid[0] if valid else candidates[0]
    print(f"Recommended     : {recommended['relative_path']}")
    print(
        "Image info      : "
        f"{recommended.get('width')} x {recommended.get('height')} / "
        f"mode={recommended.get('mode')} / "
        f"format={recommended.get('format')}"
    )

    print(f"Displayed paths : {min(len(candidates), max_display)}")
    for index, candidate in enumerate(candidates[:max_display], start=1):
        status = "PASS" if candidate["pillow_decode"] else "FAIL"
        print(
            f"{index:>2}. [{status}] {candidate['relative_path']} "
            f"| {candidate.get('width')}x{candidate.get('height')} "
            f"| mode={candidate.get('mode')}"
        )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    if args.max_display <= 0:
        raise ValueError("--max-display must be greater than 0")
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    data_raw = project_root / "data" / "raw"
    all_raw_images = _iter_image_paths(data_raw)

    # NEU-DET는 Casting 후보에서 분리한다.
    casting_images = [
        path
        for path in all_raw_images
        if "neu_det" not in {part.lower() for part in path.parts}
    ]

    casting_normal_paths = [
        path
        for path in casting_images
        if _has_named_component(path, NORMAL_DIRECTORY_NAMES)
    ]
    casting_defect_paths = [
        path
        for path in casting_images
        if _has_named_component(path, DEFECT_DIRECTORY_NAMES)
    ]

    neu_det_root = data_raw / "neu_det" / "NEU-DET"
    neu_det_paths = _iter_image_paths(neu_det_root)

    casting_normal = [
        _inspect_image(path, project_root) for path in casting_normal_paths
    ]
    casting_defect = [
        _inspect_image(path, project_root) for path in casting_defect_paths
    ]
    neu_det = [_inspect_image(path, project_root) for path in neu_det_paths]

    references = _find_text_references(project_root)

    result = {
        "project_root": str(project_root),
        "data_raw_exists": data_raw.is_dir(),
        "all_raw_image_count": len(all_raw_images),
        "casting_normal": casting_normal,
        "casting_defect": casting_defect,
        "neu_det": neu_det,
        "existing_casting_path_references": references,
        "ready_for_day10_real_analysis": bool(
            casting_normal and casting_defect and neu_det
        ),
    }

    print("=" * 100)
    print("DAY 10 - REAL IMAGE SAMPLE CANDIDATE INSPECTION")
    print("=" * 100)
    print(f"Project root         : {project_root}")
    print(f"data/raw exists      : {data_raw.is_dir()}")
    print(f"All raw image count  : {len(all_raw_images):,}")
    print(f"NEU-DET root         : {neu_det_root}")
    print(f"NEU-DET root exists  : {neu_det_root.is_dir()}")

    _print_category(
        "CASTING NORMAL CANDIDATES",
        casting_normal,
        max_display=args.max_display,
    )
    _print_category(
        "CASTING DEFECT CANDIDATES",
        casting_defect,
        max_display=args.max_display,
    )
    _print_category(
        "NEU-DET CANDIDATES",
        neu_det,
        max_display=args.max_display,
    )

    print("\n[EXISTING CASTING PATH REFERENCES]")
    print(f"Reference count : {len(references):,}")
    for reference in references[:30]:
        print(
            f"- {reference['path']}:{reference['line_number']} "
            f"| {reference['line']}"
        )
    if len(references) > 30:
        print(f"... {len(references) - 30} more reference(s) omitted")

    ready = result["ready_for_day10_real_analysis"]
    print("\n[RESULT]")
    print(f"Ready for Day 10 real analysis : {ready}")

    print("\n[JSON]")
    # 후보 전체 경로가 너무 많으므로 JSON 출력에는 각 범주의 앞 10개만 포함한다.
    compact_result = {
        **result,
        "casting_normal": casting_normal[:10],
        "casting_defect": casting_defect[:10],
        "neu_det": neu_det[:10],
    }
    print(json.dumps(compact_result, ensure_ascii=False, indent=2))

    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

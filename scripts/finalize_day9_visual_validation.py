"""Day 9 Detection Figure 육안 검증 결과를 Artifact로 저장한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


FIGURE_FILENAMES = (
    "day9_detection_class_distribution.png",
    "day9_detection_box_statistics.png",
    "day9_detection_annotation_overview.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 9 Detection Figure 육안 검증 결과 저장",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="프로젝트 루트",
    )
    parser.add_argument(
        "--status",
        choices=("pass", "fail"),
        required=True,
        help="육안 검증 최종 상태",
    )
    parser.add_argument(
        "--class-distribution-pass",
        action="store_true",
        help="Class Distribution Figure가 정상인지 확인",
    )
    parser.add_argument(
        "--box-statistics-pass",
        action="store_true",
        help="Box Statistics Figure가 정상인지 확인",
    )
    parser.add_argument(
        "--annotation-overview-pass",
        action="store_true",
        help="Annotation Overview Figure가 정상인지 확인",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="육안 검증 메모",
    )
    return parser.parse_args()


def inspect_figure(path: Path, *, project_root: Path) -> dict[str, object]:
    """PNG 존재·크기·Pillow Decode 가능 여부를 확인한다."""
    if not path.is_file():
        raise FileNotFoundError(f"Figure 파일이 없습니다: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Figure 파일이 비어 있습니다: {path}")

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Figure Decode에 실패했습니다: {path}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Figure 크기가 올바르지 않습니다: {path}")

    try:
        relative_path = path.resolve().relative_to(project_root.resolve())
        stored_path = relative_path.as_posix()
    except ValueError:
        stored_path = str(path.resolve())

    return {
        "path": stored_path,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "decode_valid": True,
    }


def build_validation_payload(
    *,
    project_root: Path,
    status: str,
    class_distribution_pass: bool,
    box_statistics_pass: bool,
    annotation_overview_pass: bool,
    notes: str,
) -> dict[str, object]:
    """수동 확인과 기계적 PNG 검증 결과를 하나의 Payload로 만든다."""
    project_root = project_root.resolve()
    figure_dir = project_root / "reports" / "figures"
    figures = [
        inspect_figure(figure_dir / filename, project_root=project_root)
        for filename in FIGURE_FILENAMES
    ]

    manual_checks = {
        "class_distribution": class_distribution_pass,
        "box_statistics": box_statistics_pass,
        "annotation_overview": annotation_overview_pass,
    }
    requested_pass = status == "pass"
    all_manual_checks_passed = all(manual_checks.values())

    if requested_pass and not all_manual_checks_passed:
        missing = [name for name, passed in manual_checks.items() if not passed]
        raise ValueError(
            "PASS 저장에는 세 Figure의 수동 확인이 모두 필요합니다: "
            + ", ".join(missing)
        )

    return {
        "day": 9,
        "validation_name": "object_detection_dataset_figures",
        "status": "PASS" if requested_pass else "FAIL",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manual_checks": manual_checks,
        "all_manual_checks_passed": all_manual_checks_passed,
        "machine_checks": {
            "required_figure_count": len(FIGURE_FILENAMES),
            "decoded_figure_count": len(figures),
            "all_files_exist": True,
            "all_files_non_empty": True,
            "all_files_decodable": True,
        },
        "figures": figures,
        "notes": notes.strip(),
    }


def save_validation_payload(
    payload: dict[str, object],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_path = (
        project_root
        / "reports"
        / "artifacts"
        / "day9_detection_visual_validation.json"
    )

    try:
        payload = build_validation_payload(
            project_root=project_root,
            status=args.status,
            class_distribution_pass=args.class_distribution_pass,
            box_statistics_pass=args.box_statistics_pass,
            annotation_overview_pass=args.annotation_overview_pass,
            notes=args.notes,
        )
        save_validation_payload(payload, output_path=output_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("=" * 100)
    print("DAY 9 - DETECTION FIGURE VISUAL VALIDATION")
    print("=" * 100)
    print(f"Status   : {payload['status']}")
    print(f"Figures  : {len(payload['figures'])}")
    print(f"Artifact : {output_path}")

    if payload["status"] != "PASS":
        print("\n[FAIL] 육안 검증이 FAIL로 기록됐습니다.")
        return 1

    print("\n[PASS] Day 9 Figure 육안 검증 결과 저장 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

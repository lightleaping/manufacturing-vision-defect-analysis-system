from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.finalize_day9_visual_validation import (
    FIGURE_FILENAMES,
    build_validation_payload,
    save_validation_payload,
)


def _create_figures(root: Path) -> None:
    figure_dir = root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for filename in FIGURE_FILENAMES:
        Image.new("RGB", (320, 240), (120, 120, 120)).save(
            figure_dir / filename
        )


def test_pass_requires_all_manual_checks(tmp_path: Path) -> None:
    _create_figures(tmp_path)
    with pytest.raises(ValueError, match="수동 확인"):
        build_validation_payload(
            project_root=tmp_path,
            status="pass",
            class_distribution_pass=True,
            box_statistics_pass=False,
            annotation_overview_pass=True,
            notes="",
        )


def test_missing_figure_is_rejected(tmp_path: Path) -> None:
    _create_figures(tmp_path)
    (tmp_path / "reports" / "figures" / FIGURE_FILENAMES[0]).unlink()
    with pytest.raises(FileNotFoundError):
        build_validation_payload(
            project_root=tmp_path,
            status="pass",
            class_distribution_pass=True,
            box_statistics_pass=True,
            annotation_overview_pass=True,
            notes="",
        )


def test_pass_payload_contains_decoded_figure_metadata(tmp_path: Path) -> None:
    _create_figures(tmp_path)
    payload = build_validation_payload(
        project_root=tmp_path,
        status="pass",
        class_distribution_pass=True,
        box_statistics_pass=True,
        annotation_overview_pass=True,
        notes="boxes aligned",
    )

    assert payload["status"] == "PASS"
    assert payload["all_manual_checks_passed"] is True
    assert len(payload["figures"]) == 3
    assert all(item["decode_valid"] for item in payload["figures"])
    assert all(item["width"] == 320 for item in payload["figures"])


def test_save_validation_payload_writes_utf8_json(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "artifacts" / "visual.json"
    saved = save_validation_payload(
        {"status": "PASS", "notes": "육안 검증 완료"},
        output_path=output,
    )
    assert saved == output
    assert json.loads(output.read_text(encoding="utf-8"))["notes"] == "육안 검증 완료"

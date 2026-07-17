from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from PIL import Image


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), (100, 100, 100)).save(path)


def test_inspect_day10_sample_candidates_reports_ready(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "data/raw/casting/ok_front/normal.jpg")
    _write_image(tmp_path / "data/raw/casting/def_front/defect.jpg")
    _write_image(
        tmp_path
        / "data/raw/neu_det/NEU-DET/train/images/crazing/crazing_1.jpg"
    )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "inspect_day10_sample_candidates.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(tmp_path),
            "--max-display",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Ready for Day 10 real analysis : True" in completed.stdout

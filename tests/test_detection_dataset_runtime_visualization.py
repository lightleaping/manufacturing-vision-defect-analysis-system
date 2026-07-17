"""Day 11 Dataset Overlay Figure 생성 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.run_day11_detection_dataset_validation import (
    run_day11_detection_dataset_validation,
)
from tests.test_detection_dataset_validation import create_three_split_project


def test_validation_script_creates_artifact_and_figures(tmp_path: Path) -> None:
    manifest_path = create_three_split_project(tmp_path)
    payload_raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Class별 대표 Figure에는 6개 Class가 필요하므로 각 누락 Class Record를
    # Train Split에 추가한다.
    train_records = payload_raw["splits"]["train"]
    dataset_root = tmp_path / "data" / "raw" / "neu_det" / "NEU-DET" / "train"
    from tests.detection_test_helpers import write_pascal_voc_xml, write_rgb_image

    existing_classes = {
        class_name
        for record in train_records
        for class_name in record["class_names"]
    }
    for class_name in (
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled_in_scale",
        "scratches",
    ):
        if class_name in existing_classes:
            continue
        record_name = f"{class_name}_extra"
        image_path = dataset_root / "images" / f"{record_name}.jpg"
        xml_path = dataset_root / "annotations" / f"{record_name}.xml"
        write_rgb_image(image_path, width=10, height=8)
        write_pascal_voc_xml(
            xml_path,
            image_filename=f"metadata_{record_name}.jpg",
            width=10,
            height=8,
            boxes=[(class_name, (2, 2, 8, 7))],
        )
        train_records.append(
            {
                "key": f"train/{record_name}",
                "image_path": f"NEU-DET\\train\\images\\{record_name}.jpg",
                "annotation_path": (
                    f"NEU-DET\\train\\annotations\\{record_name}.xml"
                ),
                "image_width": 10,
                "image_height": 8,
                "image_mode": "RGB",
                "class_names": [class_name],
                "boxes": [[2, 2, 8, 7]],
                "source_split": "train",
            }
        )
    manifest_path.write_text(json.dumps(payload_raw), encoding="utf-8")

    payload = run_day11_detection_dataset_validation(project_root=tmp_path)

    artifact_path = (
        tmp_path / "reports" / "artifacts" / "day11_detection_dataset_validation.json"
    )
    batch_figure = (
        tmp_path / "reports" / "figures" / "day11_detection_dataset_batch.png"
    )
    duplicate_figure = (
        tmp_path / "reports" / "figures" / "day11_detection_target_overlay.png"
    )
    assert payload["validation_passed"] is True
    assert artifact_path.is_file()
    assert batch_figure.is_file()
    assert duplicate_figure.is_file()
    with Image.open(batch_figure) as image:
        assert image.size == (900, 520)
    with Image.open(duplicate_figure) as image:
        assert image.width > 0 and image.height > 0

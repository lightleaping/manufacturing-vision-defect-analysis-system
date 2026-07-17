from pathlib import Path

from PIL import Image

from src.detection.dataset_analysis import analyze_detection_dataset
from src.detection.dataset_config import DetectionDatasetConfig
from src.detection.dataset_visualization import (
    create_annotation_overview_figure,
    create_box_statistics_figure,
    create_class_distribution_figure,
)


def _build_result(tmp_path: Path):
    images_dir = tmp_path / "IMAGES"
    annotations_dir = tmp_path / "ANNOTATIONS"
    images_dir.mkdir()
    annotations_dir.mkdir()

    for index, class_name in enumerate(("crazing", "scratches")):
        stem = f"sample_{index}"
        Image.new("L", (200, 200), 100 + index * 20).save(
            images_dir / f"{stem}.jpg"
        )
        (annotations_dir / f"{stem}.xml").write_text(
            f"""
<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object>
    <name>{class_name}</name>
    <bndbox>
      <xmin>20</xmin><ymin>30</ymin><xmax>150</xmax><ymax>170</ymax>
    </bndbox>
  </object>
</annotation>
""".strip(),
            encoding="utf-8",
        )

    config = DetectionDatasetConfig(
        project_root=tmp_path,
        dataset_root=tmp_path,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
        figures_dir=tmp_path / "figures",
    )
    return config, analyze_detection_dataset(config)


def _assert_valid_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.width > 0
        assert image.height > 0


def test_create_day9_figures(tmp_path: Path) -> None:
    config, result = _build_result(tmp_path)
    class_figure = create_class_distribution_figure(
        result,
        tmp_path / "class_distribution.png",
    )
    box_figure = create_box_statistics_figure(
        result,
        tmp_path / "box_statistics.png",
    )
    overview_figure = create_annotation_overview_figure(
        result,
        dataset_root=config.dataset_root,
        output_path=tmp_path / "annotation_overview.png",
        max_samples=2,
    )

    _assert_valid_png(class_figure)
    _assert_valid_png(box_figure)
    _assert_valid_png(overview_figure)

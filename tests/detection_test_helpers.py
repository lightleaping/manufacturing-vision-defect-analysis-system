"""Day 11 합성 Pascal VOC Fixture Helper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

from PIL import Image

from src.detection.detection_dataset import DetectionSample


BoxSpec = tuple[str, tuple[int, int, int, int]]


def write_rgb_image(
    path: Path,
    *,
    width: int = 10,
    height: int = 8,
    color: tuple[int, int, int] = (10, 20, 30),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=color).save(path)


def write_pascal_voc_xml(
    path: Path,
    *,
    image_filename: str,
    width: int = 10,
    height: int = 8,
    boxes: Sequence[BoxSpec],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image_filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for class_name, (xmin, ymin, xmax, ymax) in boxes:
        object_node = ET.SubElement(annotation, "object")
        ET.SubElement(object_node, "name").text = class_name
        box = ET.SubElement(object_node, "bndbox")
        ET.SubElement(box, "xmin").text = str(xmin)
        ET.SubElement(box, "ymin").text = str(ymin)
        ET.SubElement(box, "xmax").text = str(xmax)
        ET.SubElement(box, "ymax").text = str(ymax)

    ET.ElementTree(annotation).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def create_sample(
    root: Path,
    *,
    record_id: str,
    split: str = "train",
    width: int = 10,
    height: int = 8,
    boxes: Sequence[BoxSpec] = (("crazing", (1, 1, 10, 8)),),
) -> DetectionSample:
    image_path = root / "images" / f"{record_id}.jpg"
    annotation_path = root / "annotations" / f"{record_id}.xml"
    write_rgb_image(image_path, width=width, height=height)
    write_pascal_voc_xml(
        annotation_path,
        # Day 9에서 확인된 filename metadata mismatch를 재현한다.
        image_filename=f"metadata_{record_id}.jpg",
        width=width,
        height=height,
        boxes=boxes,
    )
    return DetectionSample(
        image_path=image_path,
        annotation_path=annotation_path,
        split=split,
        record_id=record_id,
    )

"""NEU-DET Pascal VOC를 Torchvision Detection Dataset으로 변환한다.

[기존 코드 참고]
Day 9 ``splits.json``의 실제 Schema와 Pascal VOC 좌표 분석 결과를 사용한다.

[신규 구현]
- Manifest의 논리 Image·XML Pair를 그대로 따른다.
- ``NEU-DET\\...`` 경로는 ``data/raw/neu_det`` 아래에서 해석한다.
- 원본 1-based inclusive 좌표를 0-based exclusive XYXY로 변환한다.
- Duplicate Box 기본 정책은 원본 보존이다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from src.detection.dataset_config import normalize_annotation_class_name
from src.detection.model_config import (
    DEFAULT_DEFECT_CLASS_NAMES,
    DuplicateBoxPolicy,
    build_detection_label_maps,
    validate_defect_class_names,
    validate_duplicate_box_policy,
    validate_split_name,
)
from src.detection.transforms import DetectionTarget, DetectionTransform


DEFAULT_NEU_DET_ROOT = Path("data/raw/neu_det")
_IMAGE_KEYS = ("image_path", "image")
_ANNOTATION_KEYS = ("annotation_path", "annotation")
_RECORD_ID_KEYS = ("key", "record_id", "id")


@dataclass(frozen=True, slots=True)
class DetectionSample:
    """한 이미지와 한 Pascal VOC XML의 논리 Pair."""

    image_path: Path
    annotation_path: Path
    split: str
    record_id: str

    def __post_init__(self) -> None:
        validate_split_name(self.split)
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be pathlib.Path.")
        if not isinstance(self.annotation_path, Path):
            raise TypeError("annotation_path must be pathlib.Path.")
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be a non-empty str.")


@dataclass(frozen=True, slots=True)
class ParsedObject:
    class_name: str
    original_box: tuple[int, int, int, int]


def _normalize_xml_class_name(raw_class_name: str) -> str:
    """Day 9의 NEU-DET Class alias 정책을 Dataset Loader에서도 재사용한다.

    실제 원본 XML에는 ``rolled-in_scale``처럼 구분자가 혼합된 표기가
    존재한다. Day 9에서 확정한 정규화 함수가 이를 canonical class인
    ``rolled_in_scale``로 바꾼다.

    지원하지 않는 Class는 여기서 임의 변환하지 않고 원문을 유지한다.
    이후 Dataset의 class mapping 검증에서 일관된 ``Unsupported class``
    오류를 발생시키기 위함이다.
    """
    stripped = raw_class_name.strip()
    try:
        return normalize_annotation_class_name(stripped)
    except (TypeError, ValueError):
        return stripped


def _read_json_object(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError("manifest_path must be pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}.")

    with path.open(mode="r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise TypeError("Manifest top-level value must be an object.")
    return payload


def _first_string(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _portable_relative_path(raw_path: str) -> Path:
    """Windows 구분자가 포함된 Manifest 경로를 현재 OS용 Path로 바꾼다."""
    windows_path = PureWindowsPath(raw_path)
    if windows_path.drive or windows_path.root:
        return Path(raw_path)
    return Path(*windows_path.parts)


def _resolve_record_path(
    *,
    project_root: Path,
    dataset_root: Path,
    raw_path: str,
) -> Path:
    """Day 9 Manifest 경로를 실제 파일 경로로 해석한다.

    실제 Manifest는 ``NEU-DET\\train\\...`` 형태이므로 첫 기준은
    ``project_root/data/raw/neu_det``이다. 이미 프로젝트 상대 경로가 들어온
    Fixture나 향후 Schema도 지원하기 위해 project_root 기준 후보도 확인한다.
    """
    raw_as_path = Path(raw_path)
    if raw_as_path.is_absolute():
        return raw_as_path.resolve()

    portable = _portable_relative_path(raw_path)
    candidates = (
        (dataset_root / portable).resolve(),
        (project_root / portable).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 존재 여부는 __getitem__에서 명확히 검증한다. 실제 Day 9 Schema의
    # 기준인 dataset_root 후보를 오류 메시지에 사용한다.
    return candidates[0]


def _extract_split_records(
    payload: Mapping[str, Any],
    split: str,
) -> list[Mapping[str, Any]]:
    splits_value = payload.get("splits")
    if not isinstance(splits_value, Mapping):
        raise KeyError("Manifest must contain a 'splits' object.")

    records = splits_value.get(split)
    if not isinstance(records, list):
        raise KeyError(f"Manifest splits must contain {split!r} list.")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError("Every split record must be an object.")
    return list(records)


def load_detection_samples_from_manifest(
    *,
    manifest_path: Path,
    project_root: Path,
    split: str,
    dataset_root: Path | None = None,
) -> list[DetectionSample]:
    """Day 9 ``splits.json``을 DetectionSample 목록으로 변환한다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    validate_split_name(split)

    resolved_project_root = project_root.resolve()
    resolved_dataset_root = (
        dataset_root.resolve()
        if dataset_root is not None
        else (resolved_project_root / DEFAULT_NEU_DET_ROOT).resolve()
    )
    payload = _read_json_object(manifest_path)
    records = _extract_split_records(payload, split)

    samples: list[DetectionSample] = []
    seen_record_ids: set[str] = set()
    for index, record in enumerate(records):
        image_value = _first_string(record, _IMAGE_KEYS)
        annotation_value = _first_string(record, _ANNOTATION_KEYS)
        record_id = _first_string(record, _RECORD_ID_KEYS)

        if image_value is None or annotation_value is None:
            raise KeyError(
                f"Manifest record {index} is missing image_path or "
                f"annotation_path. Available keys: {sorted(record.keys())}."
            )
        if record_id is None:
            raise KeyError(f"Manifest record {index} is missing key.")
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate manifest record key: {record_id!r}.")
        seen_record_ids.add(record_id)

        samples.append(
            DetectionSample(
                image_path=_resolve_record_path(
                    project_root=resolved_project_root,
                    dataset_root=resolved_dataset_root,
                    raw_path=image_value,
                ),
                annotation_path=_resolve_record_path(
                    project_root=resolved_project_root,
                    dataset_root=resolved_dataset_root,
                    raw_path=annotation_value,
                ),
                split=split,
                record_id=record_id,
            )
        )

    if not samples:
        raise ValueError(f"Manifest split {split!r} must not be empty.")
    return samples


def parse_pascal_voc_objects(
    annotation_path: Path,
) -> tuple[tuple[int, int], list[ParsedObject]]:
    """XML에서 Image 크기와 Object Class·Box를 읽는다.

    XML의 ``filename`` metadata는 Day 9에서 174건 불일치가 확인됐으므로
    Pair 결정에 사용하지 않는다. Manifest의 실제 Image·XML 경로가 기준이다.
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation does not exist: {annotation_path}.")

    try:
        root = ET.parse(annotation_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid Pascal VOC XML: {annotation_path}.") from error

    size_node = root.find("size")
    if size_node is None:
        raise ValueError("Pascal VOC XML must contain <size>.")

    def required_int(parent: ET.Element, tag: str) -> int:
        node = parent.find(tag)
        if node is None or node.text is None:
            raise ValueError(f"Pascal VOC XML is missing <{tag}>.")
        try:
            return int(node.text.strip())
        except ValueError as error:
            raise ValueError(f"<{tag}> must contain an integer.") from error

    width = required_int(size_node, "width")
    height = required_int(size_node, "height")
    if width <= 0 or height <= 0:
        raise ValueError("XML image width and height must be positive.")

    objects: list[ParsedObject] = []
    for object_index, object_node in enumerate(root.findall("object")):
        name_node = object_node.find("name")
        box_node = object_node.find("bndbox")
        if name_node is None or name_node.text is None:
            raise ValueError(f"Object {object_index} is missing <name>.")
        if box_node is None:
            raise ValueError(f"Object {object_index} is missing <bndbox>.")

        raw_class_name = name_node.text.strip()
        if not raw_class_name:
            raise ValueError(f"Object {object_index} has a blank class name.")
        class_name = _normalize_xml_class_name(raw_class_name)

        objects.append(
            ParsedObject(
                class_name=class_name,
                original_box=(
                    required_int(box_node, "xmin"),
                    required_int(box_node, "ymin"),
                    required_int(box_node, "xmax"),
                    required_int(box_node, "ymax"),
                ),
            )
        )

    return (width, height), objects


def convert_voc_box_to_torchvision(
    box: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """1-based inclusive VOC Box를 0-based exclusive XYXY로 변환한다."""
    if len(box) != 4:
        raise ValueError("Pascal VOC box must contain four coordinates.")
    xmin, ymin, xmax, ymax = box

    if xmin < 1 or ymin < 1:
        raise ValueError("Pascal VOC xmin and ymin must be at least 1.")
    if xmax > image_width or ymax > image_height:
        raise ValueError("Pascal VOC xmax/ymax exceed image dimensions.")
    if xmax < xmin or ymax < ymin:
        raise ValueError("Pascal VOC box has reversed coordinates.")

    converted = (
        float(xmin - 1),
        float(ymin - 1),
        float(xmax),
        float(ymax),
    )
    if converted[2] <= converted[0] or converted[3] <= converted[1]:
        raise ValueError("Converted box must have positive width and height.")
    return converted


def apply_duplicate_box_policy(
    objects: Sequence[ParsedObject],
    policy: DuplicateBoxPolicy,
) -> list[ParsedObject]:
    """설정에 따라 원본 보존 또는 정확히 같은 중복만 제거한다."""
    if policy == "preserve":
        return list(objects)

    unique: list[ParsedObject] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for item in objects:
        signature = (item.class_name, item.original_box)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


class NeuDetDetectionDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Torchvision Faster R-CNN 입력 계약을 만족하는 NEU-DET Dataset."""

    def __init__(
        self,
        *,
        samples: Sequence[DetectionSample],
        transform: DetectionTransform,
        class_names: Sequence[str] = DEFAULT_DEFECT_CLASS_NAMES,
        duplicate_box_policy: str = "preserve",
    ) -> None:
        if isinstance(samples, (str, bytes)):
            raise TypeError("samples must be a sequence of DetectionSample.")
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("samples must not be empty.")
        if any(not isinstance(sample, DetectionSample) for sample in self.samples):
            raise TypeError("Every sample must be DetectionSample.")
        if not callable(transform):
            raise TypeError("transform must be callable.")

        self.transform = transform
        self.class_names = validate_defect_class_names(class_names)
        self.class_to_index, self.index_to_class = build_detection_label_maps(
            self.class_names
        )
        self.duplicate_box_policy = validate_duplicate_box_policy(
            duplicate_box_policy
        )

    @classmethod
    def from_manifest(
        cls,
        *,
        manifest_path: Path,
        project_root: Path,
        split: str,
        transform: DetectionTransform,
        class_names: Sequence[str] = DEFAULT_DEFECT_CLASS_NAMES,
        duplicate_box_policy: str = "preserve",
        dataset_root: Path | None = None,
    ) -> "NeuDetDetectionDataset":
        samples = load_detection_samples_from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split=split,
            dataset_root=dataset_root,
        )
        return cls(
            samples=samples,
            transform=transform,
            class_names=class_names,
            duplicate_box_policy=duplicate_box_policy,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be int.")
        sample = self.samples[index]

        if not sample.image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {sample.image_path}.")

        try:
            with Image.open(sample.image_path) as opened:
                image = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Could not decode image: {sample.image_path}.") from error

        (xml_width, xml_height), parsed_objects = parse_pascal_voc_objects(
            sample.annotation_path
        )
        if image.width != xml_width or image.height != xml_height:
            raise ValueError(
                "Image dimensions do not match Pascal VOC XML: "
                f"image={image.size}, xml={(xml_width, xml_height)}."
            )

        parsed_objects = apply_duplicate_box_policy(
            parsed_objects,
            self.duplicate_box_policy,
        )

        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        for object_index, item in enumerate(parsed_objects):
            if item.class_name not in self.class_to_index:
                raise ValueError(
                    f"Unsupported class {item.class_name!r} at object "
                    f"{object_index}."
                )
            boxes.append(
                convert_voc_box_to_torchvision(
                    item.original_box,
                    image_width=image.width,
                    image_height=image.height,
                )
            )
            labels.append(self.class_to_index[item.class_name])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        ).to(dtype=torch.float32)

        target: DetectionTarget = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        transformed_image, transformed_target = self.transform(image, target)
        self._validate_output(transformed_image, transformed_target)
        return transformed_image, transformed_target

    def _validate_output(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> None:
        if image.dtype != torch.float32:
            raise TypeError("Detection image must use torch.float32.")
        if image.ndim != 3 or int(image.shape[0]) != 3:
            raise ValueError("Detection image must have shape [3, H, W].")
        if image.numel() == 0:
            raise ValueError("Detection image must not be empty.")
        if not torch.isfinite(image).all():
            raise ValueError("Detection image must contain only finite values.")
        if float(image.min().item()) < 0.0 or float(image.max().item()) > 1.0:
            raise ValueError("Detection image values must be in [0, 1].")

        required_keys = {"boxes", "labels", "image_id", "area", "iscrowd"}
        if set(target) != required_keys:
            raise ValueError(
                f"Unexpected target keys: {set(target)}. "
                f"Expected: {required_keys}."
            )

        boxes = target["boxes"]
        labels = target["labels"]
        image_id = target["image_id"]
        area = target["area"]
        iscrowd = target["iscrowd"]

        if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise TypeError("target['boxes'] must be FloatTensor[N, 4].")
        if labels.dtype != torch.int64 or labels.ndim != 1:
            raise TypeError("target['labels'] must be Int64Tensor[N].")
        if image_id.dtype != torch.int64 or image_id.shape != (1,):
            raise TypeError("target['image_id'] must be Int64Tensor[1].")
        if area.dtype != torch.float32 or area.ndim != 1:
            raise TypeError("target['area'] must be FloatTensor[N].")
        if iscrowd.dtype != torch.int64 or iscrowd.ndim != 1:
            raise TypeError("target['iscrowd'] must be Int64Tensor[N].")

        box_count = len(boxes)
        if not (
            box_count == len(labels) == len(area) == len(iscrowd)
        ):
            raise ValueError("Box, label, area, and iscrowd counts must match.")
        if not torch.isfinite(boxes).all() or not torch.isfinite(area).all():
            raise ValueError("Target boxes and area must contain finite values.")
        if not torch.all(iscrowd == 0):
            raise ValueError("NEU-DET iscrowd values must all be 0.")

        height = int(image.shape[-2])
        width = int(image.shape[-1])
        if boxes.numel() > 0:
            if not torch.all(boxes[:, 2] > boxes[:, 0]):
                raise ValueError("Every box must satisfy xmax > xmin.")
            if not torch.all(boxes[:, 3] > boxes[:, 1]):
                raise ValueError("Every box must satisfy ymax > ymin.")
            if not torch.all(boxes[:, 0] >= 0) or not torch.all(boxes[:, 1] >= 0):
                raise ValueError("Box minimum coordinates must be non-negative.")
            if not torch.all(boxes[:, 2] <= width) or not torch.all(
                boxes[:, 3] <= height
            ):
                raise ValueError("Box maximum coordinates exceed image bounds.")
            if not torch.all(
                (labels >= 1) & (labels <= len(self.class_names))
            ):
                raise ValueError(
                    "Detection labels must be in [1, num_defect_classes]."
                )
            expected_area = (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )
            if not torch.allclose(area, expected_area):
                raise ValueError("target['area'] does not match target['boxes'].")

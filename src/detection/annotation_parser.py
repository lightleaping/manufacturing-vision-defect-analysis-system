"""Pascal VOC XML Annotation Parser와 좌표 무결성 검증."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

from .dataset_config import normalize_annotation_class_name


@dataclass(frozen=True)
class BoundingBox:
    """원본 XML의 XYXY 좌표를 변형하지 않고 보존한다."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return float("inf")
        return self.width / self.height

    def as_list(self) -> list[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass(frozen=True)
class DetectionObject:
    class_name: str
    box: BoundingBox


@dataclass(frozen=True)
class ParsedAnnotation:
    annotation_path: Path
    filename: str | None
    width: int
    height: int
    depth: int | None
    objects: tuple[DetectionObject, ...]


@dataclass(frozen=True)
class ValidationIssue:
    """분석 Artifact에 저장할 구조화된 오류·경고."""

    code: str
    message: str
    severity: str
    object_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "object_index": self.object_index,
        }


class AnnotationParseError(ValueError):
    """XML을 신뢰 가능한 Annotation 객체로 만들 수 없을 때 발생한다."""

    def __init__(self, code: str, message: str, *, path: Path) -> None:
        super().__init__(f"[{code}] {message}: {path}")
        self.code = code
        self.path = path
        self.message = message


def _text(
    parent: ET.Element,
    tag: str,
    *,
    path: Path,
    required: bool = True,
) -> str | None:
    node = parent.find(tag)
    value = None if node is None else node.text
    if value is None or not value.strip():
        if required:
            raise AnnotationParseError(
                "missing_field",
                f"필수 XML 필드가 없습니다: {tag}",
                path=path,
            )
        return None
    return value.strip()


def _integer(
    parent: ET.Element,
    tag: str,
    *,
    path: Path,
    required: bool = True,
) -> int | None:
    raw = _text(parent, tag, path=path, required=required)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise AnnotationParseError(
            "invalid_integer",
            f"정수 필드 형식이 잘못되었습니다: {tag}={raw!r}",
            path=path,
        ) from exc


def parse_pascal_voc_annotation(
    annotation_path: Path,
    *,
    class_name_normalizer: Callable[[str], str] = (
        normalize_annotation_class_name
    ),
) -> ParsedAnnotation:
    """Pascal VOC XML을 파싱한다.

    좌표 범위 검사는 이미지 Decode 결과가 필요한 경우가 있어
    :func:`validate_annotation`에서 수행한다.
    """
    annotation_path = annotation_path.resolve()
    try:
        root = ET.parse(annotation_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise AnnotationParseError(
            "broken_xml",
            "XML을 읽거나 파싱할 수 없습니다",
            path=annotation_path,
        ) from exc

    filename = _text(root, "filename", path=annotation_path, required=False)
    size = root.find("size")
    if size is None:
        raise AnnotationParseError(
            "missing_size",
            "size 노드가 없습니다",
            path=annotation_path,
        )

    width = _integer(size, "width", path=annotation_path)
    height = _integer(size, "height", path=annotation_path)
    depth = _integer(size, "depth", path=annotation_path, required=False)
    assert width is not None
    assert height is not None

    objects: list[DetectionObject] = []
    for object_node in root.findall("object"):
        raw_class_name = _text(object_node, "name", path=annotation_path)
        assert raw_class_name is not None
        try:
            class_name = class_name_normalizer(raw_class_name)
        except ValueError as exc:
            raise AnnotationParseError(
                "unknown_class",
                str(exc),
                path=annotation_path,
            ) from exc

        box_node = object_node.find("bndbox")
        if box_node is None:
            raise AnnotationParseError(
                "missing_bndbox",
                "object/bndbox 노드가 없습니다",
                path=annotation_path,
            )

        x_min = _integer(box_node, "xmin", path=annotation_path)
        y_min = _integer(box_node, "ymin", path=annotation_path)
        x_max = _integer(box_node, "xmax", path=annotation_path)
        y_max = _integer(box_node, "ymax", path=annotation_path)
        assert None not in (x_min, y_min, x_max, y_max)

        objects.append(
            DetectionObject(
                class_name=class_name,
                box=BoundingBox(
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                ),
            )
        )

    return ParsedAnnotation(
        annotation_path=annotation_path,
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        objects=tuple(objects),
    )


def validate_annotation(
    annotation: ParsedAnnotation,
    *,
    actual_image_size: tuple[int, int] | None = None,
    actual_filename: str | None = None,
    allow_empty: bool = False,
    small_box_area_ratio_threshold: float = 0.0001,
) -> list[ValidationIssue]:
    """이미지 크기와 Bounding Box를 검증한다.

    좌표를 임의로 Clip하거나 수정하지 않고 모든 문제를 보고한다.
    """
    issues: list[ValidationIssue] = []

    if annotation.width <= 0:
        issues.append(
            ValidationIssue(
                code="invalid_width",
                message=f"Annotation width는 0보다 커야 합니다: {annotation.width}",
                severity="error",
            )
        )
    if annotation.height <= 0:
        issues.append(
            ValidationIssue(
                code="invalid_height",
                message=f"Annotation height는 0보다 커야 합니다: {annotation.height}",
                severity="error",
            )
        )

    if actual_image_size is not None:
        if actual_image_size != (annotation.width, annotation.height):
            issues.append(
                ValidationIssue(
                    code="image_size_mismatch",
                    message=(
                        "실제 이미지 크기와 XML size가 다릅니다: "
                        f"actual={actual_image_size}, "
                        f"annotation={(annotation.width, annotation.height)}"
                    ),
                    severity="error",
                )
            )

    if (
        actual_filename is not None
        and annotation.filename is not None
        and Path(annotation.filename).name != Path(actual_filename).name
    ):
        issues.append(
            ValidationIssue(
                code="filename_mismatch",
                message=(
                    "실제 이미지 파일명과 XML filename이 다릅니다: "
                    f"actual={actual_filename}, annotation={annotation.filename}"
                ),
                severity="warning",
            )
        )

    if not annotation.objects and not allow_empty:
        issues.append(
            ValidationIssue(
                code="empty_annotation",
                message="Bounding Box가 하나도 없습니다.",
                severity="error",
            )
        )

    seen_boxes: set[tuple[str, int, int, int, int]] = set()
    image_area = annotation.width * annotation.height

    for object_index, detection_object in enumerate(annotation.objects):
        box = detection_object.box
        error_codes: list[tuple[str, str]] = []

        if box.x_min >= box.x_max:
            error_codes.append(
                ("invalid_x_order", f"x_min >= x_max: {box.as_list()}")
            )
        if box.y_min >= box.y_max:
            error_codes.append(
                ("invalid_y_order", f"y_min >= y_max: {box.as_list()}")
            )
        if box.x_min < 0:
            error_codes.append(("negative_x_min", f"x_min < 0: {box.x_min}"))
        if box.y_min < 0:
            error_codes.append(("negative_y_min", f"y_min < 0: {box.y_min}"))
        if box.x_max > annotation.width:
            error_codes.append(
                (
                    "x_max_out_of_bounds",
                    f"x_max > image width: {box.x_max} > {annotation.width}",
                )
            )
        if box.y_max > annotation.height:
            error_codes.append(
                (
                    "y_max_out_of_bounds",
                    f"y_max > image height: {box.y_max} > {annotation.height}",
                )
            )
        if box.area <= 0:
            error_codes.append(("non_positive_area", f"Box area <= 0: {box.area}"))

        for code, message in error_codes:
            issues.append(
                ValidationIssue(
                    code=code,
                    message=message,
                    severity="error",
                    object_index=object_index,
                )
            )

        if not error_codes and image_area > 0:
            area_ratio = box.area / image_area
            if area_ratio < small_box_area_ratio_threshold:
                issues.append(
                    ValidationIssue(
                        code="very_small_box",
                        message=(
                            "매우 작은 Box입니다: "
                            f"area_ratio={area_ratio:.8f}, box={box.as_list()}"
                        ),
                        severity="warning",
                        object_index=object_index,
                    )
                )

        if box.x_min == 0 or box.y_min == 0:
            issues.append(
                ValidationIssue(
                    code="zero_coordinate_observed",
                    message=(
                        "0 좌표가 발견되었습니다. 전체 Dataset 분석 후 "
                        "0-based/1-based 정책을 확정해야 합니다."
                    ),
                    severity="info",
                    object_index=object_index,
                )
            )

        duplicate_key = (
            detection_object.class_name,
            box.x_min,
            box.y_min,
            box.x_max,
            box.y_max,
        )
        if duplicate_key in seen_boxes:
            issues.append(
                ValidationIssue(
                    code="duplicate_box",
                    message=f"동일 Class·좌표 Box가 중복됩니다: {duplicate_key}",
                    severity="warning",
                    object_index=object_index,
                )
            )
        seen_boxes.add(duplicate_key)

    return issues

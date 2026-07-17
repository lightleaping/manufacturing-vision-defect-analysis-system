from pathlib import Path

import pytest

from src.detection.annotation_parser import (
    AnnotationParseError,
    parse_pascal_voc_annotation,
    validate_annotation,
)


def _write_xml(
    path: Path,
    *,
    objects: str,
    width: str = "200",
    height: str = "200",
    filename: str = "sample.jpg",
) -> Path:
    path.write_text(
        f"""
<annotation>
  <filename>{filename}</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>1</depth>
  </size>
  {objects}
</annotation>
""".strip(),
        encoding="utf-8",
    )
    return path


def _object_xml(
    class_name: str = "crazing",
    xmin: str = "10",
    ymin: str = "20",
    xmax: str = "100",
    ymax: str = "120",
) -> str:
    return f"""
<object>
  <name>{class_name}</name>
  <bndbox>
    <xmin>{xmin}</xmin>
    <ymin>{ymin}</ymin>
    <xmax>{xmax}</xmax>
    <ymax>{ymax}</ymax>
  </bndbox>
</object>
""".strip()


def test_parse_valid_annotation_with_multiple_objects(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path / "sample.xml",
        objects=_object_xml("crazing") + _object_xml("scratches", "30", "40", "80", "90"),
    )
    parsed = parse_pascal_voc_annotation(path)
    assert parsed.width == 200
    assert parsed.height == 200
    assert [item.class_name for item in parsed.objects] == [
        "crazing",
        "scratches",
    ]
    assert parsed.objects[0].box.as_list() == [10, 20, 100, 120]


def test_unknown_class_is_parse_error(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path / "unknown.xml",
        objects=_object_xml("not-a-real-class"),
    )
    with pytest.raises(AnnotationParseError) as exc_info:
        parse_pascal_voc_annotation(path)
    assert exc_info.value.code == "unknown_class"


def test_missing_bndbox_is_rejected(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path / "missing.xml",
        objects="<object><name>crazing</name></object>",
    )
    with pytest.raises(AnnotationParseError) as exc_info:
        parse_pascal_voc_annotation(path)
    assert exc_info.value.code == "missing_bndbox"


def test_invalid_integer_is_rejected(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path / "invalid.xml",
        objects=_object_xml(xmin="ten"),
    )
    with pytest.raises(AnnotationParseError) as exc_info:
        parse_pascal_voc_annotation(path)
    assert exc_info.value.code == "invalid_integer"


def test_broken_xml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.xml"
    path.write_text("<annotation>", encoding="utf-8")
    with pytest.raises(AnnotationParseError) as exc_info:
        parse_pascal_voc_annotation(path)
    assert exc_info.value.code == "broken_xml"


@pytest.mark.parametrize(
    ("xmin", "ymin", "xmax", "ymax", "expected_code"),
    [
        ("100", "20", "100", "120", "invalid_x_order"),
        ("10", "120", "100", "120", "invalid_y_order"),
        ("-1", "20", "100", "120", "negative_x_min"),
        ("10", "-1", "100", "120", "negative_y_min"),
        ("10", "20", "201", "120", "x_max_out_of_bounds"),
        ("10", "20", "100", "201", "y_max_out_of_bounds"),
    ],
)
def test_validate_invalid_box_coordinates(
    tmp_path: Path,
    xmin: str,
    ymin: str,
    xmax: str,
    ymax: str,
    expected_code: str,
) -> None:
    path = _write_xml(
        tmp_path / f"{expected_code}.xml",
        objects=_object_xml(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        ),
    )
    parsed = parse_pascal_voc_annotation(path)
    issues = validate_annotation(parsed)
    assert expected_code in {issue.code for issue in issues}


def test_validate_size_mismatch_and_duplicate_box(tmp_path: Path) -> None:
    repeated = _object_xml() + _object_xml()
    path = _write_xml(tmp_path / "duplicate.xml", objects=repeated)
    parsed = parse_pascal_voc_annotation(path)
    issues = validate_annotation(
        parsed,
        actual_image_size=(300, 300),
        actual_filename="different.jpg",
    )
    codes = {issue.code for issue in issues}
    assert "image_size_mismatch" in codes
    assert "filename_mismatch" in codes
    assert "duplicate_box" in codes


def test_empty_annotation_policy(tmp_path: Path) -> None:
    path = _write_xml(tmp_path / "empty.xml", objects="")
    parsed = parse_pascal_voc_annotation(path)
    assert "empty_annotation" in {
        issue.code for issue in validate_annotation(parsed, allow_empty=False)
    }
    assert "empty_annotation" not in {
        issue.code for issue in validate_annotation(parsed, allow_empty=True)
    }

"""
Manufacturing Vision Defect Analysis System

Day 1 dataset configuration and analysis tests.

이 테스트 파일은 실제 전체 데이터셋을 매번 다시 분석하지 않고,
작은 임시 파일을 사용하여 데이터 설정과 이미지 분석 함수의
핵심 동작을 빠르게 검증한다.

테스트 대상
----------
1. 실제 Train·Test 데이터 경로
2. 정상·불량 Label 규칙
3. 지원 이미지 확장자
4. 정상 RGB 이미지 분석
5. 손상 이미지 예외 처리
6. 지원하지 않는 파일 처리

실행 명령
---------
pytest tests/test_dataset_config_and_analysis.py -v
"""

from pathlib import Path

from PIL import Image

from src.data.dataset_analysis import (
    analyze_file,
    validate_dataset_structure,
)
from src.data.dataset_config import (
    CLASS_TO_INDEX,
    DATASET_ROOT,
    INDEX_TO_CLASS_NAME,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_ROOT,
    TRAIN_ROOT,
)


def test_dataset_paths_exist() -> None:
    """
    실제 데이터셋의 루트·Train·Test 경로가 존재하는지 확인한다.

    방지하는 오류
    ------------
    dataset_config.py의 경로가 잘못 변경되어
    데이터 분석과 학습 코드가 이미지를 찾지 못하는 문제
    """

    assert DATASET_ROOT.exists()

    assert DATASET_ROOT.is_dir()

    assert TRAIN_ROOT.exists()

    assert TRAIN_ROOT.is_dir()

    assert TEST_ROOT.exists()

    assert TEST_ROOT.is_dir()


def test_dataset_structure_is_valid() -> None:
    """
    Train·Test 안에 정상·불량 클래스 폴더가 모두 있는지 확인한다.

    정상 구조
    --------
    train/
    ├── ok_front/
    └── def_front/

    test/
    ├── ok_front/
    └── def_front/

    방지하는 오류
    ------------
    클래스 폴더가 삭제되거나 이름이 변경되었는데
    이미지가 0장인 상태로 분석이 진행되는 문제
    """

    # 데이터 구조가 정상이라면 예외 없이 함수가 종료된다.
    validate_dataset_structure()


def test_class_mapping_uses_defect_as_positive_class() -> None:
    """
    정상은 0, 불량은 1로 고정되어 있는지 확인한다.

    이 기준이 중요한 이유
    ---------------------
    이후 Precision·Recall·F1 계산에서
    불량을 Positive Class로 해석하기 위해 사용한다.
    """

    assert CLASS_TO_INDEX == {
        "ok_front": 0,
        "def_front": 1,
    }

    assert INDEX_TO_CLASS_NAME == {
        0: "NORMAL",
        1: "DEFECT",
    }


def test_supported_image_extensions_include_common_formats() -> None:
    """
    프로젝트에서 허용할 주요 이미지 확장자가
    설정에 포함되어 있는지 확인한다.

    방지하는 오류
    ------------
    확장자 설정이 실수로 변경되어
    정상 이미지가 미지원 파일로 처리되는 문제
    """

    assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS

    assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS

    assert ".png" in SUPPORTED_IMAGE_EXTENSIONS


def test_analyze_file_returns_valid_rgb_image_information(
    tmp_path: Path,
) -> None:
    """
    정상 RGB 이미지를 분석했을 때
    크기·모드·채널·상태가 올바르게 반환되는지 확인한다.

    입력
    ----
    테스트 중 임시로 생성한 32 × 24 RGB JPEG 이미지

    예상 출력
    --------
    width=32

    height=24

    image_mode=RGB

    channel_count=3

    status=VALID
    """

    image_path = (
        tmp_path
        / "valid_rgb_image.jpeg"
    )

    test_image = Image.new(
        mode="RGB",
        size=(32, 24),
        color=(
            120,
            130,
            140,
        ),
    )

    test_image.save(
        image_path,
        format="JPEG",
    )

    result = analyze_file(
        file_path=image_path,
        split_name="train",
        source_class_name="ok_front",
    )

    assert result["split"] == "train"

    assert (
        result["source_class"]
        == "ok_front"
    )

    assert result["class_index"] == 0

    assert result["class_name"] == "NORMAL"

    assert result["extension"] == ".jpeg"

    assert (
        result[
            "is_supported_extension"
        ]
        is True
    )

    assert (
        result["is_valid_image"]
        is True
    )

    assert (
        result["is_corrupted"]
        is False
    )

    assert result["width"] == 32

    assert result["height"] == 24

    assert result["image_mode"] == "RGB"

    assert result["channel_count"] == 3

    assert result["status"] == "VALID"

    assert result["error_message"] is None


def test_analyze_file_marks_broken_image_as_corrupted(
    tmp_path: Path,
) -> None:
    """
    확장자는 JPEG이지만 실제 내용이 이미지가 아닌 파일을
    손상 이미지로 처리하는지 확인한다.

    입력
    ----
    파일명:
    broken_image.jpeg

    실제 내용:
    일반 문자열

    예상 출력
    --------
    is_valid_image=False

    is_corrupted=True

    status=CORRUPTED
    """

    broken_image_path = (
        tmp_path
        / "broken_image.jpeg"
    )

    broken_image_path.write_text(
        "this is not a real image",
        encoding="utf-8",
    )

    result = analyze_file(
        file_path=broken_image_path,
        split_name="test",
        source_class_name="def_front",
    )

    assert result["split"] == "test"

    assert (
        result["source_class"]
        == "def_front"
    )

    assert result["class_index"] == 1

    assert result["class_name"] == "DEFECT"

    assert (
        result[
            "is_supported_extension"
        ]
        is True
    )

    assert (
        result["is_valid_image"]
        is False
    )

    assert (
        result["is_corrupted"]
        is True
    )

    assert result["status"] == "CORRUPTED"

    assert (
        result["error_message"]
        is not None
    )


def test_analyze_file_marks_text_file_as_unsupported(
    tmp_path: Path,
) -> None:
    """
    지원하지 않는 TXT 파일을
    손상 이미지와 구분하여 처리하는지 확인한다.

    중요 구분
    --------
    CORRUPTED

    → 이미지 확장자이지만 파일 내용이 손상됨


    UNSUPPORTED_EXTENSION

    → 프로젝트에서 허용하지 않는 파일 형식
    """

    text_file_path = (
        tmp_path
        / "readme.txt"
    )

    text_file_path.write_text(
        "not an image",
        encoding="utf-8",
    )

    result = analyze_file(
        file_path=text_file_path,
        split_name="train",
        source_class_name="ok_front",
    )

    assert result["extension"] == ".txt"

    assert (
        result[
            "is_supported_extension"
        ]
        is False
    )

    assert (
        result["is_valid_image"]
        is False
    )

    # 지원하지 않는 파일은
    # 손상 이미지와 별도 상태로 구분한다.
    assert (
        result["is_corrupted"]
        is False
    )

    assert (
        result["status"]
        == "UNSUPPORTED_EXTENSION"
    )

    assert (
        result["error_message"]
        == "지원하지 않는 파일 확장자입니다."
    )
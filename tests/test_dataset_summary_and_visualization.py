"""
Manufacturing Vision Defect Analysis System

Day 1 dataset summary and visualization tests.

이 테스트 파일은 데이터 분석 요약과 시각화 함수가
작은 임시 데이터에서도 올바르게 동작하는지 검증한다.

테스트 대상
----------
1. 데이터셋 요약 통계
2. Boolean·문자열 유효 이미지 처리
3. 재현 가능한 샘플 이미지 선택
4. 클래스 분포 그래프 생성
5. 정상·불량 샘플 이미지 Grid 생성
6. 분석 CSV 누락 예외 처리

실행 명령
---------
python -m pytest \
    tests/test_dataset_summary_and_visualization.py \
    -v
"""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.dataset_analysis import (
    create_dataset_summary,
)
import src.data.dataset_visualization as dataset_visualization


def create_test_analysis_dataframe() -> pd.DataFrame:
    """
    데이터 요약 테스트에 사용할 작은 DataFrame을 생성한다.

    구성
    ----
    Train NORMAL:
    정상 이미지 2장

    Train DEFECT:
    정상 이미지 2장

    Test NORMAL:
    정상 이미지 1장

    Test DEFECT:
    손상 이미지 1장

    전체:
    6개 파일
    """

    return pd.DataFrame(
        [
            {
                "split": "train",
                "source_class": "ok_front",
                "class_index": 0,
                "class_name": "NORMAL",
                "file_path": (
                    "data/train/ok_front/"
                    "normal_1.jpeg"
                ),
                "file_name": "normal_1.jpeg",
                "extension": ".jpeg",
                "file_size_bytes": 100,
                "is_supported_extension": True,
                "is_valid_image": True,
                "is_corrupted": False,
                "width": 300,
                "height": 300,
                "image_mode": "RGB",
                "channel_count": 3,
                "status": "VALID",
                "error_message": None,
            },
            {
                "split": "train",
                "source_class": "ok_front",
                "class_index": 0,
                "class_name": "NORMAL",
                "file_path": (
                    "data/train/ok_front/"
                    "normal_2.jpeg"
                ),
                "file_name": "normal_2.jpeg",
                "extension": ".jpeg",
                "file_size_bytes": 110,
                "is_supported_extension": True,
                "is_valid_image": True,
                "is_corrupted": False,
                "width": 300,
                "height": 300,
                "image_mode": "RGB",
                "channel_count": 3,
                "status": "VALID",
                "error_message": None,
            },
            {
                "split": "train",
                "source_class": "def_front",
                "class_index": 1,
                "class_name": "DEFECT",
                "file_path": (
                    "data/train/def_front/"
                    "defect_1.jpeg"
                ),
                "file_name": "defect_1.jpeg",
                "extension": ".jpeg",
                "file_size_bytes": 120,
                "is_supported_extension": True,
                "is_valid_image": True,
                "is_corrupted": False,
                "width": 300,
                "height": 300,
                "image_mode": "RGB",
                "channel_count": 3,
                "status": "VALID",
                "error_message": None,
            },
            {
                "split": "train",
                "source_class": "def_front",
                "class_index": 1,
                "class_name": "DEFECT",
                "file_path": (
                    "data/train/def_front/"
                    "defect_2.jpeg"
                ),
                "file_name": "defect_2.jpeg",
                "extension": ".jpeg",
                "file_size_bytes": 130,
                "is_supported_extension": True,
                "is_valid_image": True,
                "is_corrupted": False,
                "width": 300,
                "height": 300,
                "image_mode": "RGB",
                "channel_count": 3,
                "status": "VALID",
                "error_message": None,
            },
            {
                "split": "test",
                "source_class": "ok_front",
                "class_index": 0,
                "class_name": "NORMAL",
                "file_path": (
                    "data/test/ok_front/"
                    "normal_test.jpeg"
                ),
                "file_name": "normal_test.jpeg",
                "extension": ".jpeg",
                "file_size_bytes": 140,
                "is_supported_extension": True,
                "is_valid_image": True,
                "is_corrupted": False,
                "width": 300,
                "height": 300,
                "image_mode": "RGB",
                "channel_count": 3,
                "status": "VALID",
                "error_message": None,
            },
            {
                "split": "test",
                "source_class": "def_front",
                "class_index": 1,
                "class_name": "DEFECT",
                "file_path": (
                    "data/test/def_front/"
                    "broken.jpeg"
                ),
                "file_name": "broken.jpeg",
                "extension": ".jpeg",
                "file_size_bytes": 10,
                "is_supported_extension": True,
                "is_valid_image": False,
                "is_corrupted": True,
                "width": None,
                "height": None,
                "image_mode": None,
                "channel_count": None,
                "status": "CORRUPTED",
                "error_message": (
                    "UnidentifiedImageError"
                ),
            },
        ]
    )


def create_rgb_image(
    image_path: Path,
    color: tuple[int, int, int],
) -> None:
    """
    시각화 테스트용 RGB 이미지를 생성한다.

    Parameters
    ----------
    image_path:
        저장할 이미지 경로

    color:
        RGB 색상
    """

    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.new(
        mode="RGB",
        size=(32, 32),
        color=color,
    )

    image.save(
        image_path,
        format="JPEG",
    )


def test_create_dataset_summary_returns_correct_counts() -> None:
    """
    이미지별 결과를 요약했을 때
    전체·유효·손상 이미지 수가 정확한지 확인한다.

    방지하는 오류
    ------------
    전체 파일 수와 유효 이미지 수를 혼동하거나
    손상 이미지가 정상 이미지 수에 포함되는 문제
    """

    analysis_dataframe = (
        create_test_analysis_dataframe()
    )

    summary = create_dataset_summary(
        analysis_dataframe
    )

    assert summary[
        "total_file_count"
    ] == 6

    assert summary[
        "valid_image_count"
    ] == 5

    assert summary[
        "corrupted_image_count"
    ] == 1

    assert summary[
        "unsupported_file_count"
    ] == 0


def test_create_dataset_summary_returns_image_properties() -> None:
    """
    이미지 크기와 RGB 채널 정보가
    올바르게 집계되는지 확인한다.

    현재 가상 데이터
    ----------------
    정상 이미지:
    5장

    크기:
    300 × 300

    Mode:
    RGB

    Channel:
    3
    """

    analysis_dataframe = (
        create_test_analysis_dataframe()
    )

    summary = create_dataset_summary(
        analysis_dataframe
    )

    assert (
        summary[
            "image_size_counts"
        ]
        == [
            {
                "width": 300,
                "height": 300,
                "image_count": 5,
            }
        ]
    )

    assert (
        summary[
            "image_mode_counts"
        ]
        == [
            {
                "image_mode": "RGB",
                "channel_count": 3,
                "image_count": 5,
            }
        ]
    )


def test_create_valid_image_mask_supports_boolean_values() -> None:
    """
    CSV의 is_valid_image 열이 실제 Boolean일 때
    유효 이미지 Mask를 올바르게 생성하는지 확인한다.
    """

    dataframe = pd.DataFrame(
        {
            "is_valid_image": [
                True,
                False,
                True,
            ]
        }
    )

    valid_mask = (
        dataset_visualization
        .create_valid_image_mask(
            dataframe
        )
    )

    assert valid_mask.tolist() == [
        True,
        False,
        True,
    ]


def test_create_valid_image_mask_supports_string_values() -> None:
    """
    CSV 환경에 따라 True·False가 문자열로 읽혀도
    올바르게 처리되는지 확인한다.

    방지하는 오류
    ------------
    문자열 "False"가 Python에서
    참처럼 해석되어 손상 이미지가 포함되는 문제
    """

    dataframe = pd.DataFrame(
        {
            "is_valid_image": [
                "True",
                "False",
                "TRUE",
            ]
        }
    )

    valid_mask = (
        dataset_visualization
        .create_valid_image_mask(
            dataframe
        )
    )

    assert valid_mask.tolist() == [
        True,
        False,
        True,
    ]


def test_select_sample_paths_is_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    같은 데이터와 Random Seed에서
    항상 같은 샘플을 선택하는지 확인한다.

    재현성이 필요한 이유
    -------------------
    실행할 때마다 README 결과 이미지가 달라지면
    결과 비교와 문서 관리가 어려워진다.
    """

    project_root = (
        tmp_path
        / "test_project"
    )

    relative_paths: list[str] = []

    for image_index in range(6):

        relative_path = (
            Path("data")
            / "train"
            / "ok_front"
            / f"normal_{image_index}.jpeg"
        )

        absolute_path = (
            project_root
            / relative_path
        )

        create_rgb_image(
            image_path=absolute_path,
            color=(
                image_index * 20,
                100,
                150,
            ),
        )

        relative_paths.append(
            relative_path.as_posix()
        )

    dataframe = pd.DataFrame(
        {
            "split": [
                "train"
            ] * 6,
            "class_name": [
                "NORMAL"
            ] * 6,
            "file_path": relative_paths,
            "is_valid_image": [
                True
            ] * 6,
        }
    )

    monkeypatch.setattr(
        dataset_visualization,
        "PROJECT_ROOT",
        project_root,
    )

    first_selection = (
        dataset_visualization
        .select_sample_paths(
            analysis_dataframe=dataframe,
            class_name="NORMAL",
            sample_count=4,
        )
    )

    second_selection = (
        dataset_visualization
        .select_sample_paths(
            analysis_dataframe=dataframe,
            class_name="NORMAL",
            sample_count=4,
        )
    )

    assert (
        first_selection
        == second_selection
    )

    assert len(
        first_selection
    ) == 4

    assert all(
        image_path.exists()
        for image_path
        in first_selection
    )


def test_create_class_distribution_chart_creates_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    작은 분석 DataFrame으로
    클래스 분포 PNG를 생성할 수 있는지 확인한다.
    """

    output_directory = (
        tmp_path
        / "artifacts"
    )

    output_path = (
        output_directory
        / "class_distribution.png"
    )

    monkeypatch.setattr(
        dataset_visualization,
        "OUTPUT_DIRECTORY",
        output_directory,
    )

    monkeypatch.setattr(
        dataset_visualization,
        "CLASS_DISTRIBUTION_PATH",
        output_path,
    )

    dataframe = (
        create_test_analysis_dataframe()
    )

    dataset_visualization \
        .create_class_distribution_chart(
            analysis_dataframe=dataframe
        )

    assert output_path.exists()

    assert output_path.is_file()

    assert (
        output_path.stat().st_size
        > 0
    )


def test_create_sample_image_grid_creates_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    임시 NORMAL·DEFECT 이미지로
    샘플 Grid PNG를 생성할 수 있는지 확인한다.

    실제 프로젝트 파일을 수정하지 않도록
    pytest의 tmp_path를 사용한다.
    """

    project_root = (
        tmp_path
        / "test_project"
    )

    output_directory = (
        project_root
        / "reports"
        / "artifacts"
    )

    sample_output_path = (
        output_directory
        / "sample_images.png"
    )

    records: list[
        dict[str, object]
    ] = []

    class_settings = [
        (
            "NORMAL",
            "ok_front",
            (
                100,
                150,
                200,
            ),
        ),
        (
            "DEFECT",
            "def_front",
            (
                200,
                100,
                100,
            ),
        ),
    ]

    for (
        class_name,
        source_class_name,
        base_color,
    ) in class_settings:

        for image_index in range(4):

            relative_path = (
                Path("data")
                / "train"
                / source_class_name
                / (
                    f"{class_name.lower()}"
                    f"_{image_index}.jpeg"
                )
            )

            absolute_path = (
                project_root
                / relative_path
            )

            create_rgb_image(
                image_path=absolute_path,
                color=(
                    min(
                        base_color[0]
                        + image_index * 5,
                        255,
                    ),
                    base_color[1],
                    base_color[2],
                ),
            )

            records.append(
                {
                    "split": "train",
                    "class_name": (
                        class_name
                    ),
                    "file_path": (
                        relative_path
                        .as_posix()
                    ),
                    "is_valid_image": True,
                }
            )

    dataframe = pd.DataFrame(
        records
    )

    monkeypatch.setattr(
        dataset_visualization,
        "PROJECT_ROOT",
        project_root,
    )

    monkeypatch.setattr(
        dataset_visualization,
        "OUTPUT_DIRECTORY",
        output_directory,
    )

    monkeypatch.setattr(
        dataset_visualization,
        "SAMPLE_IMAGES_PATH",
        sample_output_path,
    )

    dataset_visualization \
        .create_sample_image_grid(
            analysis_dataframe=dataframe
        )

    assert sample_output_path.exists()

    assert sample_output_path.is_file()

    assert (
        sample_output_path.stat().st_size
        > 0
    )


def test_load_analysis_dataframe_raises_error_when_csv_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    분석 CSV가 없을 때
    명확한 FileNotFoundError를 발생시키는지 확인한다.

    방지하는 오류
    ------------
    분석 단계를 실행하지 않았는데
    pandas 내부 오류만 출력되어
    문제 원인을 알기 어려워지는 상황
    """

    missing_csv_path = (
        tmp_path
        / "missing_analysis.csv"
    )

    monkeypatch.setattr(
        dataset_visualization,
        "ANALYSIS_CSV_PATH",
        missing_csv_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Day 1 데이터 분석 CSV가 "
            "존재하지 않습니다"
        ),
    ):

        dataset_visualization \
            .load_analysis_dataframe()
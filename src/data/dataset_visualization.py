"""
Manufacturing Vision Defect Analysis System

Day 1 dataset visualization module.

이 모듈은 dataset_analysis.py가 생성한 CSV 결과를 이용하여
클래스 분포 그래프와 정상·불량 샘플 이미지를 생성한다.

주요 역할
---------
1. Day 1 이미지 분석 CSV 불러오기
2. Train·Test별 NORMAL·DEFECT 이미지 수 시각화
3. Train 데이터에서 정상·불량 샘플 이미지 선택
4. 샘플 이미지 Grid 생성
5. 결과 이미지를 reports/artifacts에 저장

전체 호출 흐름
-------------
python -m src.data.dataset_visualization

→ main()

→ load_analysis_dataframe()

→ create_class_distribution_chart()

→ create_sample_image_grid()

→ PNG 결과 저장
"""

from __future__ import annotations

import random
from pathlib import Path

# 화면이 없는 실행 환경에서도
# PNG 파일을 생성할 수 있도록 Agg Backend를 사용한다.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset_config import (
    PROJECT_ROOT,
)


# -----------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------

# dataset_analysis.py가 생성한 이미지별 상세 분석 CSV이다.
ANALYSIS_CSV_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day1_dataset_analysis.csv"
)


# 시각화 결과를 저장할 폴더이다.
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
)


# Train·Test별 클래스 수를 비교하는 그래프이다.
CLASS_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "day1_class_distribution.png"
)


# 정상·불량 이미지의 실제 모습을 비교하는 샘플 Grid이다.
SAMPLE_IMAGES_PATH = (
    OUTPUT_DIRECTORY
    / "day1_sample_images.png"
)


# 같은 코드를 다시 실행해도
# 동일한 샘플 이미지를 선택하기 위한 Seed이다.
RANDOM_SEED = 42


# 클래스마다 표시할 이미지 수이다.
SAMPLES_PER_CLASS = 4


# -----------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------

def load_analysis_dataframe() -> pd.DataFrame:
    """
    Day 1 이미지 분석 CSV를 불러온다.

    Returns
    -------
    pandas.DataFrame
        이미지별 데이터 분석 결과

    Raises
    ------
    FileNotFoundError
        분석 CSV가 존재하지 않을 때 발생한다.

    ValueError
        CSV가 비어 있거나 필수 열이 없을 때 발생한다.

    이 검사가 필요한 이유
    --------------------
    시각화 코드는 dataset_analysis.py의 결과를 입력으로 사용한다.

    분석을 실행하지 않은 상태에서 시각화를 실행하면
    원인을 알기 어려운 pandas 오류가 발생할 수 있다.

    따라서 입력 파일과 필수 열을 먼저 명확하게 검사한다.
    """

    if not ANALYSIS_CSV_PATH.exists():

        raise FileNotFoundError(
            "Day 1 데이터 분석 CSV가 존재하지 않습니다. "
            "먼저 다음 명령을 실행하세요: "
            "python -m src.data.dataset_analysis"
        )

    analysis_dataframe = pd.read_csv(
        ANALYSIS_CSV_PATH
    )

    if analysis_dataframe.empty:

        raise ValueError(
            "Day 1 데이터 분석 CSV가 비어 있습니다."
        )

    required_columns = {
        "split",
        "class_name",
        "class_index",
        "file_path",
        "is_valid_image",
    }

    missing_columns = (
        required_columns
        - set(
            analysis_dataframe.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "데이터 분석 CSV에 필수 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    return analysis_dataframe


# -----------------------------------------------------------------
# Valid-image filtering
# -----------------------------------------------------------------

def create_valid_image_mask(
    analysis_dataframe: pd.DataFrame,
) -> pd.Series:
    """
    유효 이미지 행을 선택하기 위한 Boolean Mask를 만든다.

    Parameters
    ----------
    analysis_dataframe:
        이미지별 분석 결과

    Returns
    -------
    pandas.Series
        유효 이미지이면 True,
        유효하지 않으면 False

    별도 함수가 필요한 이유
    ----------------------
    CSV를 읽는 환경에 따라 Boolean 값이

    True·False

    또는

    "True"·"False"

    문자열로 해석될 가능성을 함께 처리하기 위해 사용한다.
    """

    valid_image_values = (
        analysis_dataframe[
            "is_valid_image"
        ]
    )

    if (
        valid_image_values.dtype
        == bool
    ):

        return valid_image_values

    return (
        valid_image_values
        .astype(str)
        .str.lower()
        .eq("true")
    )


# -----------------------------------------------------------------
# Class distribution chart
# -----------------------------------------------------------------

def create_class_distribution_chart(
    analysis_dataframe: pd.DataFrame,
) -> None:
    """
    Train·Test별 NORMAL·DEFECT 이미지 수를
    Grouped Bar Chart로 저장한다.

    Parameters
    ----------
    analysis_dataframe:
        이미지별 분석 결과

    Output
    ------
    reports/artifacts/
    day1_class_distribution.png
    """

    valid_image_mask = (
        create_valid_image_mask(
            analysis_dataframe
        )
    )

    valid_images = (
        analysis_dataframe[
            valid_image_mask
        ]
        .copy()
    )

    # 행:
    # train, test
    #
    # 열:
    # NORMAL, DEFECT
    #
    # 값:
    # 이미지 개수
    class_counts = (
        valid_images
        .groupby(
            [
                "split",
                "class_name",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    # 그래프 순서를 항상 동일하게 유지한다.
    split_order = [
        "train",
        "test",
    ]

    class_order = [
        "NORMAL",
        "DEFECT",
    ]

    class_counts = (
        class_counts
        .reindex(
            index=split_order,
            columns=class_order,
            fill_value=0,
        )
    )

    normal_counts = (
        class_counts[
            "NORMAL"
        ]
        .to_numpy()
    )

    defect_counts = (
        class_counts[
            "DEFECT"
        ]
        .to_numpy()
    )

    x_positions = np.arange(
        len(
            split_order
        )
    )

    bar_width = 0.36

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    normal_bars = axis.bar(
        x_positions
        - bar_width / 2,
        normal_counts,
        width=bar_width,
        label="NORMAL",
    )

    defect_bars = axis.bar(
        x_positions
        + bar_width / 2,
        defect_counts,
        width=bar_width,
        label="DEFECT",
    )

    # 각 막대 위에 실제 이미지 수를 표시한다.
    axis.bar_label(
        normal_bars,
        padding=3,
    )

    axis.bar_label(
        defect_bars,
        padding=3,
    )

    axis.set_title(
        "Casting Product Dataset "
        "Class Distribution"
    )

    axis.set_xlabel(
        "Dataset Split"
    )

    axis.set_ylabel(
        "Number of Images"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            "Train",
            "Test",
        ]
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        CLASS_DISTRIBUTION_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    # 여러 번 실행할 때 Matplotlib Figure가
    # 메모리에 계속 남지 않도록 닫는다.
    plt.close(
        figure
    )


# -----------------------------------------------------------------
# Sample selection
# -----------------------------------------------------------------

def select_sample_paths(
    analysis_dataframe: pd.DataFrame,
    class_name: str,
    sample_count: int,
) -> list[Path]:
    """
    Train 데이터에서 특정 클래스의 샘플 이미지를 선택한다.

    Parameters
    ----------
    analysis_dataframe:
        이미지별 분석 결과

    class_name:
        선택할 프로젝트 클래스 이름

        가능한 값:
        - NORMAL
        - DEFECT

    sample_count:
        선택할 이미지 수

    Returns
    -------
    list[pathlib.Path]
        선택한 이미지의 절대 경로 목록

    Raises
    ------
    ValueError
        선택할 수 있는 유효 이미지가 없을 때 발생한다.

    재현성
    ------
    random.Random(42)를 사용하므로
    같은 데이터에서 다시 실행해도
    동일한 샘플 이미지가 선택된다.
    """

    valid_image_mask = (
        create_valid_image_mask(
            analysis_dataframe
        )
    )

    selected_rows = (
        analysis_dataframe[
            valid_image_mask
            & (
                analysis_dataframe[
                    "split"
                ]
                == "train"
            )
            & (
                analysis_dataframe[
                    "class_name"
                ]
                == class_name
            )
        ]
        .copy()
    )

    relative_paths = sorted(
        selected_rows[
            "file_path"
        ]
        .astype(str)
        .tolist()
    )

    if not relative_paths:

        raise ValueError(
            "샘플 이미지가 없습니다: "
            f"class_name={class_name}"
        )

    actual_sample_count = min(
        sample_count,
        len(
            relative_paths
        ),
    )

    # 클래스마다 독립적으로 같은 Seed를 사용한다.
    #
    # 정렬된 파일 목록을 대상으로 선택하므로
    # 실행 순서가 달라져도 결과를 재현할 수 있다.
    random_generator = random.Random(
        RANDOM_SEED
    )

    selected_relative_paths = (
        random_generator.sample(
            relative_paths,
            k=actual_sample_count,
        )
    )

    selected_absolute_paths = [
        (
            PROJECT_ROOT
            / relative_path
        )
        for relative_path
        in selected_relative_paths
    ]

    return selected_absolute_paths


# -----------------------------------------------------------------
# Sample image visualization
# -----------------------------------------------------------------

def create_sample_image_grid(
    analysis_dataframe: pd.DataFrame,
) -> None:
    """
    NORMAL·DEFECT Train 샘플 이미지를
    2행 Grid로 저장한다.

    첫 번째 행
    -----------
    NORMAL 이미지 4장

    두 번째 행
    -----------
    DEFECT 이미지 4장

    Output
    ------
    reports/artifacts/
    day1_sample_images.png
    """

    normal_sample_paths = (
        select_sample_paths(
            analysis_dataframe=(
                analysis_dataframe
            ),
            class_name="NORMAL",
            sample_count=(
                SAMPLES_PER_CLASS
            ),
        )
    )

    defect_sample_paths = (
        select_sample_paths(
            analysis_dataframe=(
                analysis_dataframe
            ),
            class_name="DEFECT",
            sample_count=(
                SAMPLES_PER_CLASS
            ),
        )
    )

    sample_rows = [
        (
            "NORMAL",
            normal_sample_paths,
        ),
        (
            "DEFECT",
            defect_sample_paths,
        ),
    ]

    figure, axes = plt.subplots(
        nrows=2,
        ncols=SAMPLES_PER_CLASS,
        figsize=(14, 7),
        squeeze=False,
    )

    for row_index, (
        class_name,
        sample_paths,
    ) in enumerate(
        sample_rows
    ):

        for column_index in range(
            SAMPLES_PER_CLASS
        ):

            axis = axes[
                row_index
            ][
                column_index
            ]

            # 데이터 수가 요청한 샘플 수보다 적더라도
            # 빈 칸에서 오류가 발생하지 않도록 처리한다.
            if (
                column_index
                >= len(
                    sample_paths
                )
            ):

                axis.axis(
                    "off"
                )

                continue

            image_path = (
                sample_paths[
                    column_index
                ]
            )

            # 모든 시각화 이미지를 RGB로 통일한다.
            #
            # 현재 데이터는 이미 RGB이지만,
            # 이후 다른 이미지가 추가되더라도
            # 동일한 표시 규칙을 유지할 수 있다.
            with Image.open(
                image_path
            ) as image:

                rgb_image = (
                    image
                    .convert("RGB")
                    .copy()
                )

            axis.imshow(
                rgb_image
            )

            axis.set_title(
                f"{class_name} "
                f"Sample "
                f"{column_index + 1}"
            )

            axis.axis(
                "off"
            )

    figure.suptitle(
        "Casting Product "
        "Train Image Samples",
        fontsize=16,
    )

    figure.tight_layout()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        SAMPLE_IMAGES_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# -----------------------------------------------------------------
# Console output
# -----------------------------------------------------------------

def print_visualization_result() -> None:
    """
    생성한 시각화 결과 경로를 PowerShell에 출력한다.
    """

    print()
    print("=" * 88)

    print(
        "DAY 1 - DATASET VISUALIZATION"
    )

    print("=" * 88)

    print()
    print("[OUTPUT FILES]")

    print(
        "class_distribution:",
        CLASS_DISTRIBUTION_PATH
        .relative_to(
            PROJECT_ROOT
        )
        .as_posix(),
    )

    print(
        "sample_images     :",
        SAMPLE_IMAGES_PATH
        .relative_to(
            PROJECT_ROOT
        )
        .as_posix(),
    )

    print()
    print("[VALIDATION]")

    print(
        "class_distribution_exists:",
        CLASS_DISTRIBUTION_PATH.exists(),
    )

    print(
        "sample_images_exists     :",
        SAMPLE_IMAGES_PATH.exists(),
    )

    print()
    print("=" * 88)


# -----------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------

def main() -> None:
    """
    Day 1 데이터 시각화 전체 흐름을 실행한다.

    실행 명령
    --------
    python -m src.data.dataset_visualization
    """

    analysis_dataframe = (
        load_analysis_dataframe()
    )

    create_class_distribution_chart(
        analysis_dataframe=(
            analysis_dataframe
        )
    )

    create_sample_image_grid(
        analysis_dataframe=(
            analysis_dataframe
        )
    )

    print_visualization_result()


if __name__ == "__main__":
    main()
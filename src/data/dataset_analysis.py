"""
Manufacturing Vision Defect Analysis System

Dataset analysis module.

이 모듈은 Casting Product 이미지 데이터셋을 실제로 열어
학습 전에 확인해야 하는 기본 데이터 품질 정보를 분석한다.

주요 역할
---------
1. Train·Test 클래스 폴더 확인
2. 데이터셋의 모든 파일 수집
3. 이미지 확장자 확인
4. Pillow를 이용한 손상 이미지 검사
5. 이미지 Width·Height 확인
6. 이미지 Mode 확인
7. 이미지 채널 수 확인
8. 이미지별 상세 결과를 CSV로 저장
9. 전체 요약 결과를 JSON으로 저장

전체 호출 흐름
-------------
python -m src.data.dataset_analysis

→ main()

→ analyze_dataset()

→ analyze_split()

→ analyze_file()

→ create_dataset_summary()

→ save_analysis_results()

→ CSV·JSON 결과 저장
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.data.dataset_config import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS_NAME,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_ROOT,
    TRAIN_ROOT,
)


# -----------------------------------------------------------------
# Analysis output paths
# -----------------------------------------------------------------

# Day 1 분석 결과를 저장할 폴더이다.
#
# 최종 생성 파일:
#
# reports/
# └── artifacts/
#     ├── day1_dataset_analysis.csv
#     └── day1_dataset_summary.json
ANALYSIS_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
)


# 이미지 한 장마다 분석한 상세 결과를 저장한다.
ANALYSIS_CSV_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "day1_dataset_analysis.csv"
)


# 전체 이미지 개수, 클래스 분포, 이미지 크기 등의
# 요약 정보를 JSON 형식으로 저장한다.
ANALYSIS_JSON_PATH = (
    ANALYSIS_OUTPUT_DIR
    / "day1_dataset_summary.json"
)


# -----------------------------------------------------------------
# Path helper
# -----------------------------------------------------------------

def convert_to_project_relative_path(
    file_path: Path,
) -> str:
    """
    절대 경로를 프로젝트 기준 상대 경로로 변환한다.

    Parameters
    ----------
    file_path:
        변환할 파일의 절대 또는 상대 Path 객체

    Returns
    -------
    str
        프로젝트 루트 기준 상대 경로

    Example
    -------
    입력:

    C:/Users/kflow/Downloads/
    manufacturing-vision-defect-analysis-system/
    data/raw/sample.jpg

    출력:

    data/raw/sample.jpg

    이 함수가 필요한 이유
    --------------------
    분석 CSV에 사용자 컴퓨터의 전체 절대 경로를 저장하면
    다른 컴퓨터에서는 같은 경로를 사용할 수 없다.

    프로젝트 기준 상대 경로를 저장하면
    GitHub README와 분석 결과에서 경로를 이해하기 쉽고,
    프로젝트를 다른 위치로 이동해도 구조를 유지할 수 있다.
    """

    resolved_path = file_path.resolve()

    try:
        relative_path = resolved_path.relative_to(
            PROJECT_ROOT.resolve()
        )

        # Windows의 역슬래시 대신 슬래시를 사용한다.
        #
        # 예:
        # data\\raw\\image.jpg
        #
        # →
        #
        # data/raw/image.jpg
        return relative_path.as_posix()

    except ValueError:
        # 프로젝트 외부 경로가 전달된 경우에는
        # 상대 경로 변환이 불가능하므로 원래 경로를 반환한다.
        return resolved_path.as_posix()


# -----------------------------------------------------------------
# Dataset validation
# -----------------------------------------------------------------

def validate_dataset_structure() -> None:
    """
    데이터 분석 전에 필수 폴더 구조가 존재하는지 검사한다.

    Raises
    ------
    FileNotFoundError
        Train 또는 Test 폴더가 존재하지 않을 때 발생한다.

    ValueError
        정상·불량 클래스 폴더가 존재하지 않을 때 발생한다.

    검사 대상
    --------
    train/
    ├── ok_front/
    └── def_front/

    test/
    ├── ok_front/
    └── def_front/

    이 검사가 필요한 이유
    --------------------
    데이터 경로가 잘못되어 있는데 분석을 계속하면
    이미지가 0장인 결과가 만들어질 수 있다.

    분석 시작 전에 명확한 오류를 발생시키면
    데이터 경로 문제를 빠르게 찾을 수 있다.
    """

    split_roots = {
        "train": TRAIN_ROOT,
        "test": TEST_ROOT,
    }

    for split_name, split_root in split_roots.items():

        if not split_root.exists():
            raise FileNotFoundError(
                f"{split_name} 데이터 폴더가 존재하지 않습니다: "
                f"{split_root}"
            )

        if not split_root.is_dir():
            raise NotADirectoryError(
                f"{split_name} 경로가 폴더가 아닙니다: "
                f"{split_root}"
            )

        for source_class_name in CLASS_TO_INDEX:

            class_directory = (
                split_root
                / source_class_name
            )

            if not class_directory.exists():
                raise ValueError(
                    "필수 클래스 폴더가 존재하지 않습니다: "
                    f"{class_directory}"
                )

            if not class_directory.is_dir():
                raise NotADirectoryError(
                    "클래스 경로가 폴더가 아닙니다: "
                    f"{class_directory}"
                )


# -----------------------------------------------------------------
# Single-file analysis
# -----------------------------------------------------------------

def analyze_file(
    file_path: Path,
    split_name: str,
    source_class_name: str,
) -> dict[str, Any]:
    """
    파일 한 개의 확장자·이미지 크기·채널·손상 여부를 분석한다.

    Parameters
    ----------
    file_path:
        분석할 파일 경로

    split_name:
        데이터 구분

        가능한 값:
        - train
        - test

    source_class_name:
        원본 데이터셋 클래스 폴더 이름

        가능한 값:
        - ok_front
        - def_front

    Returns
    -------
    dict[str, Any]
        이미지 한 장의 분석 결과

    정상 이미지 출력 예
    ------------------
    {
        "split": "train",
        "source_class": "def_front",
        "class_index": 1,
        "class_name": "DEFECT",
        "file_path": "...",
        "file_name": "...jpg",
        "extension": ".jpg",
        "file_size_bytes": 12345,
        "is_supported_extension": True,
        "is_valid_image": True,
        "is_corrupted": False,
        "width": 300,
        "height": 300,
        "image_mode": "L",
        "channel_count": 1,
        "status": "VALID",
        "error_message": None
    }

    처리 순서
    --------
    파일 확장자 확인

    → 지원 확장자인지 확인

    → Pillow Image.verify()

    → 이미지 파일 구조 검증

    → 이미지를 다시 열기

    → Width·Height·Mode·Channel 확인

    → 결과 반환
    """

    extension = file_path.suffix.lower()

    class_index = CLASS_TO_INDEX[
        source_class_name
    ]

    class_name = INDEX_TO_CLASS_NAME[
        class_index
    ]

    is_supported_extension = (
        extension
        in SUPPORTED_IMAGE_EXTENSIONS
    )

    # 모든 결과가 같은 열 구조를 가지도록
    # 먼저 기본값을 만든다.
    result: dict[str, Any] = {
        "split": split_name,
        "source_class": source_class_name,
        "class_index": class_index,
        "class_name": class_name,
        "file_path": convert_to_project_relative_path(
            file_path
        ),
        "file_name": file_path.name,
        "extension": extension,
        "file_size_bytes": file_path.stat().st_size,
        "is_supported_extension": (
            is_supported_extension
        ),
        "is_valid_image": False,
        "is_corrupted": False,
        "width": None,
        "height": None,
        "image_mode": None,
        "channel_count": None,
        "status": None,
        "error_message": None,
    }

    # 지원하지 않는 확장자는 Pillow 분석을 수행하지 않는다.
    #
    # 지원하지 않는 파일과 손상된 이미지는
    # 서로 다른 문제이므로 상태도 구분한다.
    if not is_supported_extension:

        result["status"] = (
            "UNSUPPORTED_EXTENSION"
        )

        result["error_message"] = (
            "지원하지 않는 파일 확장자입니다."
        )

        return result

    try:
        # 첫 번째 Image.open()
        #
        # verify()는 이미지 파일 구조가 정상인지 검사한다.
        #
        # 이미지 전체를 실제 학습 Tensor로 변환하지 않기 때문에
        # 데이터 품질 검사에 사용할 수 있다.
        with Image.open(file_path) as image:

            image.verify()

        # verify() 실행 후에는 기존 Image 객체를
        # 다시 사용할 수 없다.
        #
        # 따라서 Width, Height, Mode, Channel 정보를 얻기 위해
        # 같은 이미지를 다시 연다.
        with Image.open(file_path) as image:

            # load()는 실제 이미지 데이터를 읽는다.
            #
            # 파일 헤더만 정상이고 실제 픽셀 데이터가 손상된 경우도
            # 확인하기 위해 호출한다.
            image.load()

            width, height = image.size

            image_mode = image.mode

            # getbands() 예:
            #
            # L
            # → ("L",)
            # → 1채널
            #
            # RGB
            # → ("R", "G", "B")
            # → 3채널
            #
            # RGBA
            # → ("R", "G", "B", "A")
            # → 4채널
            channel_count = len(
                image.getbands()
            )

        result["is_valid_image"] = True

        result["is_corrupted"] = False

        result["width"] = width

        result["height"] = height

        result["image_mode"] = image_mode

        result["channel_count"] = channel_count

        result["status"] = "VALID"

        return result

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        # Pillow가 이미지를 인식하지 못하거나,
        # 이미지 데이터가 손상되었거나,
        # 파일을 정상적으로 읽지 못했을 때 처리한다.

        result["is_valid_image"] = False

        result["is_corrupted"] = True

        result["status"] = "CORRUPTED"

        result["error_message"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        return result


# -----------------------------------------------------------------
# Split analysis
# -----------------------------------------------------------------

def analyze_split(
    split_name: str,
    split_root: Path,
) -> list[dict[str, Any]]:
    """
    Train 또는 Test 폴더 하나를 분석한다.

    Parameters
    ----------
    split_name:
        분석할 데이터 구분

        예:
        - train
        - test

    split_root:
        분석할 데이터 폴더

    Returns
    -------
    list[dict[str, Any]]
        파일별 분석 결과 목록

    호출 관계
    --------
    analyze_dataset()

    → analyze_split()

    → analyze_file()
    """

    analysis_results: list[
        dict[str, Any]
    ] = []

    # CLASS_TO_INDEX에 정의한 클래스만 분석한다.
    #
    # 폴더 자동 탐색에 의존하지 않기 때문에
    # 실수로 다른 폴더가 추가되어도
    # Label 규칙이 자동으로 바뀌지 않는다.
    for source_class_name in CLASS_TO_INDEX:

        class_directory = (
            split_root
            / source_class_name
        )

        # 현재 클래스 폴더의 모든 파일을 가져온다.
        #
        # 지원하지 않는 확장자도 데이터 품질 문제일 수 있으므로
        # 처음부터 이미지 확장자만 필터링하지 않는다.
        file_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file()
        )

        for file_path in file_paths:

            file_result = analyze_file(
                file_path=file_path,
                split_name=split_name,
                source_class_name=(
                    source_class_name
                ),
            )

            analysis_results.append(
                file_result
            )

    return analysis_results


# -----------------------------------------------------------------
# Entire dataset analysis
# -----------------------------------------------------------------

def analyze_dataset() -> pd.DataFrame:
    """
    Train과 Test 전체 데이터셋을 분석한다.

    Returns
    -------
    pandas.DataFrame
        이미지별 분석 결과

    처리 흐름
    --------
    데이터 폴더 구조 검증

    → Train 분석

    → Test 분석

    → 분석 결과 결합

    → DataFrame 생성

    → 정렬

    → 결과 반환
    """

    validate_dataset_structure()

    train_results = analyze_split(
        split_name="train",
        split_root=TRAIN_ROOT,
    )

    test_results = analyze_split(
        split_name="test",
        split_root=TEST_ROOT,
    )

    all_results = (
        train_results
        + test_results
    )

    analysis_dataframe = pd.DataFrame(
        all_results
    )

    if analysis_dataframe.empty:

        raise ValueError(
            "분석할 데이터 파일을 찾지 못했습니다."
        )

    # 결과 순서를 일정하게 유지한다.
    #
    # 같은 데이터를 다시 분석했을 때
    # CSV 행 순서가 달라지는 것을 방지한다.
    analysis_dataframe = (
        analysis_dataframe
        .sort_values(
            by=[
                "split",
                "class_index",
                "file_path",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return analysis_dataframe


# -----------------------------------------------------------------
# Summary creation
# -----------------------------------------------------------------

def create_dataset_summary(
    analysis_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    이미지별 상세 분석 결과를 전체 요약 정보로 변환한다.

    Parameters
    ----------
    analysis_dataframe:
        analyze_dataset()이 반환한 DataFrame

    Returns
    -------
    dict[str, Any]
        JSON 저장이 가능한 데이터셋 요약 정보
    """

    total_file_count = int(
        len(
            analysis_dataframe
        )
    )

    valid_image_count = int(
        analysis_dataframe[
            "is_valid_image"
        ].sum()
    )

    corrupted_image_count = int(
        analysis_dataframe[
            "is_corrupted"
        ].sum()
    )

    unsupported_file_count = int(
        (
            ~analysis_dataframe[
                "is_supported_extension"
            ]
        ).sum()
    )

    # Split·Class별 파일 수
    split_class_counts = (
        analysis_dataframe
        .groupby(
            [
                "split",
                "class_name",
                "class_index",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="file_count"
        )
    )

    split_class_counts[
        "file_count"
    ] = (
        split_class_counts[
            "file_count"
        ]
        .astype(int)
    )

    # 확장자별 파일 수
    extension_counts = (
        analysis_dataframe
        .groupby(
            "extension",
            dropna=False,
        )
        .size()
        .reset_index(
            name="file_count"
        )
    )

    extension_counts[
        "file_count"
    ] = (
        extension_counts[
            "file_count"
        ]
        .astype(int)
    )

    # 정상적으로 읽은 이미지만
    # Width·Height 분석에 사용한다.
    valid_images = (
        analysis_dataframe[
            analysis_dataframe[
                "is_valid_image"
            ]
        ]
        .copy()
    )

    image_size_counts = (
        valid_images
        .groupby(
            [
                "width",
                "height",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="image_count"
        )
    )

    if not image_size_counts.empty:

        image_size_counts[
            "width"
        ] = (
            image_size_counts[
                "width"
            ]
            .astype(int)
        )

        image_size_counts[
            "height"
        ] = (
            image_size_counts[
                "height"
            ]
            .astype(int)
        )

        image_size_counts[
            "image_count"
        ] = (
            image_size_counts[
                "image_count"
            ]
            .astype(int)
        )

    # 이미지 Mode·채널 수 분석
    image_mode_counts = (
        valid_images
        .groupby(
            [
                "image_mode",
                "channel_count",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="image_count"
        )
    )

    if not image_mode_counts.empty:

        image_mode_counts[
            "channel_count"
        ] = (
            image_mode_counts[
                "channel_count"
            ]
            .astype(int)
        )

        image_mode_counts[
            "image_count"
        ] = (
            image_mode_counts[
                "image_count"
            ]
            .astype(int)
        )

    corrupted_files = (
        analysis_dataframe[
            analysis_dataframe[
                "is_corrupted"
            ]
        ][
            [
                "file_path",
                "status",
                "error_message",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    unsupported_files = (
        analysis_dataframe[
            ~analysis_dataframe[
                "is_supported_extension"
            ]
        ][
            [
                "file_path",
                "extension",
                "status",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    summary: dict[str, Any] = {

        "dataset_name": (
            "Casting Product Image Data "
            "for Quality Inspection"
        ),

        "project_class_definition": {
            "0": "NORMAL",
            "1": "DEFECT",
        },

        "dataset_paths": {
            "train": (
                convert_to_project_relative_path(
                    TRAIN_ROOT
                )
            ),
            "test": (
                convert_to_project_relative_path(
                    TEST_ROOT
                )
            ),
        },

        "total_file_count": (
            total_file_count
        ),

        "valid_image_count": (
            valid_image_count
        ),

        "corrupted_image_count": (
            corrupted_image_count
        ),

        "unsupported_file_count": (
            unsupported_file_count
        ),

        "split_class_counts": (
            split_class_counts
            .to_dict(
                orient="records"
            )
        ),

        "extension_counts": (
            extension_counts
            .to_dict(
                orient="records"
            )
        ),

        "image_size_counts": (
            image_size_counts
            .to_dict(
                orient="records"
            )
        ),

        "image_mode_counts": (
            image_mode_counts
            .to_dict(
                orient="records"
            )
        ),

        "corrupted_files": (
            corrupted_files
        ),

        "unsupported_files": (
            unsupported_files
        ),
    }

    return summary


# -----------------------------------------------------------------
# Result saving
# -----------------------------------------------------------------

def save_analysis_results(
    analysis_dataframe: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """
    데이터 분석 결과를 CSV와 JSON 파일로 저장한다.

    Parameters
    ----------
    analysis_dataframe:
        이미지별 상세 분석 결과

    summary:
        전체 데이터셋 요약 결과

    Output
    ------
    reports/artifacts/day1_dataset_analysis.csv

    reports/artifacts/day1_dataset_summary.json
    """

    # 저장 폴더가 없으면 자동으로 생성한다.
    ANALYSIS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_dataframe.to_csv(
        ANALYSIS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # utf-8-sig를 사용하는 이유
    # -------------------------
    # Windows Excel에서 CSV를 열 때
    # 한글이 깨질 가능성을 줄이기 위해 사용한다.

    with ANALYSIS_JSON_PATH.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )


# -----------------------------------------------------------------
# Console output
# -----------------------------------------------------------------

def print_analysis_summary(
    summary: dict[str, Any],
) -> None:
    """
    주요 데이터 분석 결과를 PowerShell에 출력한다.
    """

    print()
    print("=" * 88)

    print(
        "DAY 1 - IMAGE DATASET ANALYSIS"
    )

    print("=" * 88)

    print()
    print("[DATASET]")

    print(
        "name                  :",
        summary["dataset_name"],
    )

    print(
        "total_file_count      :",
        summary["total_file_count"],
    )

    print(
        "valid_image_count     :",
        summary["valid_image_count"],
    )

    print(
        "corrupted_image_count :",
        summary[
            "corrupted_image_count"
        ],
    )

    print(
        "unsupported_file_count:",
        summary[
            "unsupported_file_count"
        ],
    )

    print()
    print("[SPLIT AND CLASS COUNTS]")

    for item in summary[
        "split_class_counts"
    ]:

        print(
            f"{item['split']:5} | "
            f"{item['class_name']:6} | "
            f"label={item['class_index']} | "
            f"count={item['file_count']}"
        )

    print()
    print("[IMAGE EXTENSIONS]")

    for item in summary[
        "extension_counts"
    ]:

        print(
            f"{item['extension']}: "
            f"{item['file_count']}"
        )

    print()
    print("[IMAGE SIZES]")

    for item in summary[
        "image_size_counts"
    ]:

        print(
            f"{item['width']} x "
            f"{item['height']}: "
            f"{item['image_count']}"
        )

    print()
    print("[IMAGE MODES AND CHANNELS]")

    for item in summary[
        "image_mode_counts"
    ]:

        print(
            f"mode={item['image_mode']}, "
            f"channels="
            f"{item['channel_count']}: "
            f"{item['image_count']}"
        )

    print()
    print("[OUTPUT FILES]")

    print(
        "csv :",
        convert_to_project_relative_path(
            ANALYSIS_CSV_PATH
        ),
    )

    print(
        "json:",
        convert_to_project_relative_path(
            ANALYSIS_JSON_PATH
        ),
    )

    print()
    print("=" * 88)


# -----------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------

def main() -> None:
    """
    데이터 분석 전체 흐름을 실행한다.

    실행 명령
    --------
    python -m src.data.dataset_analysis
    """

    analysis_dataframe = (
        analyze_dataset()
    )

    summary = create_dataset_summary(
        analysis_dataframe=(
            analysis_dataframe
        )
    )

    save_analysis_results(
        analysis_dataframe=(
            analysis_dataframe
        ),
        summary=summary,
    )

    print_analysis_summary(
        summary=summary
    )


if __name__ == "__main__":
    main()
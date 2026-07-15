"""
Train·Validation 이미지 Sample 분리 모듈.

이 모듈의 책임
----------------
1. Day 1에서 확정한 Train 데이터 경로를 읽는다.
2. 이미지 파일 경로와 클래스 Label을 연결한다.
3. 전체 Train 데이터를 새로운 Train·Validation으로 분리한다.
4. Stratified Split을 사용하여 클래스 비율을 최대한 유지한다.
5. Train과 Validation에 같은 이미지 경로가 들어가지 않았는지 검증한다.
6. 분리된 데이터의 전체 수와 클래스별 개수를 출력한다.

중요
----
이 단계에서는 실제 이미지 파일을 열거나 Tensor로 변환하지 않는다.

현재는 다음 정보만 관리한다.

    이미지 파일 경로 + Label

실제 이미지 로드, RGB 변환, Transform 적용은 이후 Custom Dataset의
__getitem__()에서 구현한다.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.dataset_config import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS_NAME,
    SUPPORTED_IMAGE_EXTENSIONS,
    TRAIN_ROOT,
)


# 기존 Train 데이터 중 Validation으로 사용할 비율이다.
#
# 0.20은 전체 Train 데이터의 20%를 Validation으로 사용한다는 의미다.
#
# 현재 기존 Train 데이터:
#     6,633장
#
# 예상 분리:
#     새로운 Train: 약 5,306장
#     Validation: 약 1,327장
VALIDATION_RATIO: float = 0.20


# Train·Validation 분리 결과를 실행할 때마다 동일하게 유지하기 위한 Seed다.
#
# 동일한 이미지 목록, 동일한 Validation 비율, 동일한 Seed를 사용하면
# 같은 이미지가 같은 Split에 배정된다.
#
# 이를 통해 코드 수정 전후의 모델 성능을 더 공정하게 비교할 수 있다.
RANDOM_SEED: int = 42


@dataclass(frozen=True)
class ImageSample:
    """
    이미지 한 장의 경로와 Label을 저장하는 데이터 구조.

    Attributes
    ----------
    image_path:
        실제 이미지 파일의 전체 Path다.

        현재 단계에서는 이미지 파일을 열지 않고 Path만 저장한다.

        실제 이미지 로드는 이후 Custom Dataset의 __getitem__()에서
        Pillow의 Image.open()을 사용하여 수행한다.

    label:
        모델 학습에 사용할 정수 Label이다.

        현재 프로젝트 기준:

            0 → NORMAL
            1 → DEFECT

    frozen=True를 사용하는 이유
    ---------------------------
    ImageSample을 생성한 뒤 image_path 또는 label이 실수로 변경되지 않도록
    불변 객체로 만든다.

    Train·Validation 분리 이후 Sample의 Label이 변경되면 데이터 신뢰성이
    훼손될 수 있으므로 수정할 필요가 없는 데이터 구조로 관리한다.
    """

    image_path: Path
    label: int


def normalize_image_extensions(
    supported_extensions: Collection[str],
) -> set[str]:
    """
    이미지 확장자를 비교하기 쉬운 형식으로 정규화한다.

    Parameters
    ----------
    supported_extensions:
        프로젝트에서 지원하는 이미지 확장자 목록이다.

        예:

            {".jpg", ".jpeg", ".png"}

        또는 점이 없는 형식도 입력할 수 있다.

            {"jpg", "jpeg", "png"}

    Returns
    -------
    set[str]
        모든 확장자를 소문자이며 점으로 시작하는 형식으로 반환한다.

        예:

            {
                ".jpg",
                ".jpeg",
                ".png",
            }

    필요한 이유
    -----------
    파일 확장자는 대소문자가 다를 수 있다.

        image.jpeg
        image.JPEG

    두 파일을 같은 JPEG 이미지로 처리하기 위해 확장자를 소문자로
    변환한다.
    """

    normalized_extensions: set[str] = set()

    for extension in supported_extensions:
        # 확장자의 앞뒤 공백을 제거하고 소문자로 변환한다.
        normalized_extension = extension.strip().lower()

        # 빈 문자열은 유효한 확장자가 아니므로 무시한다.
        if not normalized_extension:
            continue

        # ".jpeg"처럼 점으로 시작하지 않으면 점을 추가한다.
        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        normalized_extensions.add(normalized_extension)

    # 지원 확장자가 하나도 남지 않으면 이미지 수집을 수행할 수 없다.
    if not normalized_extensions:
        raise ValueError(
            "지원 이미지 확장자가 비어 있습니다."
        )

    return normalized_extensions


def collect_image_samples(
    data_root: Path,
    class_to_index: Mapping[str, int] = CLASS_TO_INDEX,
    supported_extensions: Collection[str] = (
        SUPPORTED_IMAGE_EXTENSIONS
    ),
) -> list[ImageSample]:
    """
    클래스 디렉터리에서 이미지 경로와 Label을 수집한다.

    Parameters
    ----------
    data_root:
        이미지 클래스 디렉터리가 존재하는 상위 경로다.

        현재 프로젝트에서는 다음 경로가 전달된다.

            TRAIN_ROOT

        실제 구조:

            train/
            ├── def_front/
            └── ok_front/

    class_to_index:
        클래스 디렉터리 이름을 정수 Label로 변환하는 Mapping이다.

        현재 프로젝트:

            {
                "ok_front": 0,
                "def_front": 1,
            }

    supported_extensions:
        학습 데이터로 수집할 이미지 확장자 목록이다.

    Returns
    -------
    list[ImageSample]
        이미지 경로와 Label이 연결된 Sample 목록이다.

        예:

            [
                ImageSample(
                    image_path=Path(
                        ".../ok_front/image_1.jpeg"
                    ),
                    label=0,
                ),
                ImageSample(
                    image_path=Path(
                        ".../def_front/image_2.jpeg"
                    ),
                    label=1,
                ),
            ]

    Raises
    ------
    FileNotFoundError
        데이터 Root 또는 필수 클래스 디렉터리가 존재하지 않을 때 발생한다.

    NotADirectoryError
        입력 Path가 디렉터리가 아닐 때 발생한다.

    ValueError
        클래스 Mapping이 비어 있거나 수집된 이미지가 없을 때 발생한다.

    호출 관계
    ---------
    main()

        → collect_image_samples()

        → split_train_validation_samples()

        → print_split_summary()
    """

    # 전달받은 Path가 실제로 존재하는지 먼저 확인한다.
    if not data_root.exists():
        raise FileNotFoundError(
            "데이터 Root를 찾을 수 없습니다: "
            f"{data_root}"
        )

    # Path가 파일이면 클래스 하위 폴더를 탐색할 수 없으므로 예외 처리한다.
    if not data_root.is_dir():
        raise NotADirectoryError(
            "데이터 Root는 디렉터리여야 합니다: "
            f"{data_root}"
        )

    # 클래스 Mapping이 비어 있으면 폴더 이름을 Label로 변환할 수 없다.
    if not class_to_index:
        raise ValueError(
            "class_to_index가 비어 있습니다."
        )

    normalized_extensions = normalize_image_extensions(
        supported_extensions=supported_extensions,
    )

    image_samples: list[ImageSample] = []

    # Label 순서대로 클래스를 탐색한다.
    #
    # 현재 기준:
    #     ok_front  → 0
    #     def_front → 1
    #
    # 정렬된 순서로 수집하면 파일 목록의 순서도 재현하기 쉬워진다.
    sorted_class_items = sorted(
        class_to_index.items(),
        key=lambda item: item[1],
    )

    for class_directory_name, label in sorted_class_items:
        class_directory = (
            data_root
            / class_directory_name
        )

        # 프로젝트에서 필수로 정의한 클래스 폴더가 없으면
        # 조용히 건너뛰지 않고 즉시 오류를 발생시킨다.
        #
        # 예:
        #     train/ok_front가 실수로 삭제된 경우
        if not class_directory.exists():
            raise FileNotFoundError(
                "클래스 디렉터리를 찾을 수 없습니다: "
                f"{class_directory}"
            )

        if not class_directory.is_dir():
            raise NotADirectoryError(
                "클래스 경로는 디렉터리여야 합니다: "
                f"{class_directory}"
            )

        # rglob("*")은 현재 클래스 폴더와 모든 하위 폴더를 탐색한다.
        #
        # 현재 데이터는 한 단계 구조지만, 향후 하위 폴더가 추가되어도
        # 이미지 파일을 수집할 수 있도록 재귀 탐색을 사용한다.
        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in normalized_extensions
            )
        )

        # 클래스 폴더는 존재하지만 지원 이미지가 하나도 없다면
        # 잘못된 데이터 구조일 가능성이 있으므로 오류를 발생시킨다.
        if not image_paths:
            raise ValueError(
                "지원 이미지 파일이 없는 클래스입니다: "
                f"{class_directory}"
            )

        # 이미지 파일 하나마다 Path와 Label을 연결한다.
        for image_path in image_paths:
            image_samples.append(
                ImageSample(
                    image_path=image_path,
                    label=label,
                )
            )

    # 전체 클래스 탐색 후에도 이미지가 없다면 이후 분리를 수행할 수 없다.
    if not image_samples:
        raise ValueError(
            "수집된 이미지 Sample이 없습니다."
        )

    return image_samples


def validate_split_arguments(
    samples: Sequence[ImageSample],
    validation_ratio: float,
) -> None:
    """
    Stratified Split을 실행하기 전에 입력 조건을 검증한다.

    Parameters
    ----------
    samples:
        분리할 전체 이미지 Sample이다.

    validation_ratio:
        Validation으로 사용할 비율이다.

        유효 범위:

            0 < validation_ratio < 1

    Raises
    ------
    ValueError
        Sample이 없거나 Validation 비율이 잘못되었거나,
        Stratified Split을 수행하기에 클래스 Sample 수가 부족할 때 발생한다.

    필요한 이유
    -----------
    잘못된 입력을 train_test_split()에 바로 전달하면 scikit-learn 내부
    오류 메시지만 확인하게 될 수 있다.

    프로젝트 코드에서 먼저 조건을 검증하면 문제의 원인을 더 명확하게
    설명할 수 있다.
    """

    if not samples:
        raise ValueError(
            "분리할 이미지 Sample이 없습니다."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio는 0보다 크고 "
            "1보다 작아야 합니다: "
            f"{validation_ratio}"
        )

    label_counts = Counter(
        sample.label
        for sample in samples
    )

    # 이진 분류 프로젝트에서는 최소 두 개의 클래스가 필요하다.
    if len(label_counts) < 2:
        raise ValueError(
            "Stratified Split에는 최소 두 개의 "
            "클래스가 필요합니다."
        )

    # Stratified Split은 각 클래스를 Train과 Validation에 모두
    # 배치해야 하므로 클래스마다 최소 두 개 이상의 Sample이 필요하다.
    labels_with_too_few_samples = {
        label: count
        for label, count in label_counts.items()
        if count < 2
    }

    if labels_with_too_few_samples:
        raise ValueError(
            "클래스별 Sample이 너무 적습니다: "
            f"{labels_with_too_few_samples}"
        )

    validation_count = ceil(
        len(samples)
        * validation_ratio
    )

    train_count = (
        len(samples)
        - validation_count
    )

    class_count = len(label_counts)

    # Train 또는 Validation의 전체 Sample 수가 클래스 수보다 작으면
    # 모든 클래스를 각 Split에 포함할 수 없다.
    if validation_count < class_count:
        raise ValueError(
            "Validation Sample 수가 클래스 수보다 적습니다. "
            f"validation_count={validation_count}, "
            f"class_count={class_count}"
        )

    if train_count < class_count:
        raise ValueError(
            "Train Sample 수가 클래스 수보다 적습니다. "
            f"train_count={train_count}, "
            f"class_count={class_count}"
        )


def create_normalized_path_key(
    image_path: Path,
) -> str:
    """
    중복 Path 비교를 위한 표준 문자열 Key를 생성한다.

    Parameters
    ----------
    image_path:
        비교할 이미지 파일 Path다.

    Returns
    -------
    str
        절대 경로를 대소문자 구분 없이 비교할 수 있도록 변환한 문자열이다.

    Windows 환경 고려
    -----------------
    Windows 파일 시스템은 일반적으로 Path 대소문자를 구분하지 않는다.

    예:

        C:/DATA/image.jpeg

        c:/data/image.jpeg

    위 Path를 동일한 파일로 비교하기 위해 casefold()를 사용한다.
    """

    return str(
        image_path.resolve()
    ).casefold()


def validate_no_sample_overlap(
    train_samples: Sequence[ImageSample],
    validation_samples: Sequence[ImageSample],
) -> None:
    """
    Train과 Validation에 같은 이미지 Path가 존재하는지 검사한다.

    Parameters
    ----------
    train_samples:
        Train으로 분리된 이미지 Sample이다.

    validation_samples:
        Validation으로 분리된 이미지 Sample이다.

    Raises
    ------
    ValueError
        같은 이미지 Path가 Train과 Validation에 동시에 존재할 때 발생한다.

    데이터 누수 관점
    ----------------
    동일 이미지가 Train과 Validation에 동시에 존재하면 모델이 학습 중 본
    이미지를 Validation에서도 평가할 수 있다.

    그러면 Validation 성능이 실제 일반화 성능보다 높게 측정될 수 있다.
    """

    train_path_keys = {
        create_normalized_path_key(
            sample.image_path
        )
        for sample in train_samples
    }

    validation_path_keys = {
        create_normalized_path_key(
            sample.image_path
        )
        for sample in validation_samples
    }

    overlapping_path_keys = (
        train_path_keys
        & validation_path_keys
    )

    if overlapping_path_keys:
        example_paths = sorted(
            overlapping_path_keys
        )[:5]

        raise ValueError(
            "Train과 Validation에 중복 이미지가 있습니다: "
            f"{example_paths}"
        )


def split_train_validation_samples(
    samples: Sequence[ImageSample],
    validation_ratio: float = VALIDATION_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[
    list[ImageSample],
    list[ImageSample],
]:
    """
    전체 Train Sample을 새로운 Train과 Validation으로 분리한다.

    Parameters
    ----------
    samples:
        기존 Train 디렉터리에서 수집한 전체 이미지 Sample이다.

    validation_ratio:
        Validation으로 사용할 비율이다.

        기본값:

            0.20

    random_seed:
        분리 결과를 재현하기 위한 Random Seed다.

        기본값:

            42

    Returns
    -------
    tuple[list[ImageSample], list[ImageSample]]
        첫 번째 값:

            Train Sample 목록

        두 번째 값:

            Validation Sample 목록

    분리 방식
    ---------
    train_test_split()에 다음 설정을 사용한다.

        shuffle=True

            → 분리 전에 Sample 순서를 섞는다.

        stratify=labels

            → 기존 클래스 비율을 Train과 Validation에서
              최대한 비슷하게 유지한다.

        random_state=random_seed

            → 동일한 조건에서 같은 분리 결과를 재현한다.

    중요
    ----
    기존 casting_data/test 데이터는 이 함수에 전달하지 않는다.

    기존 Test 715장은 최종 모델 평가용으로 그대로 보존한다.
    """

    validate_split_arguments(
        samples=samples,
        validation_ratio=validation_ratio,
    )

    # Stratified Split에 사용할 Label 목록을 만든다.
    labels = [
        sample.label
        for sample in samples
    ]

    train_samples, validation_samples = (
        train_test_split(
            list(samples),
            test_size=validation_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=labels,
        )
    )

    # 반환 순서를 일정하게 유지하면 테스트 결과와 출력 결과를
    # 비교하기 쉽다.
    #
    # 실제 모델 학습 시 데이터 순서를 섞는 역할은 이후 DataLoader의
    # shuffle=True가 담당한다.
    train_samples = sorted(
        train_samples,
        key=lambda sample: str(
            sample.image_path
        ).casefold(),
    )

    validation_samples = sorted(
        validation_samples,
        key=lambda sample: str(
            sample.image_path
        ).casefold(),
    )

    validate_no_sample_overlap(
        train_samples=train_samples,
        validation_samples=validation_samples,
    )

    # 분리 후 전체 개수가 원본 개수와 같은지 검증한다.
    split_total = (
        len(train_samples)
        + len(validation_samples)
    )

    if split_total != len(samples):
        raise ValueError(
            "Train·Validation 분리 후 전체 Sample 수가 "
            "원본과 다릅니다. "
            f"original={len(samples)}, "
            f"split_total={split_total}"
        )

    return (
        train_samples,
        validation_samples,
    )


def count_samples_by_label(
    samples: Sequence[ImageSample],
) -> dict[int, int]:
    """
    이미지 Sample의 클래스별 개수를 계산한다.

    Parameters
    ----------
    samples:
        개수를 계산할 이미지 Sample 목록이다.

    Returns
    -------
    dict[int, int]
        Label별 Sample 수다.

        예:

            {
                0: 2300,
                1: 3006,
            }
    """

    label_counter = Counter(
        sample.label
        for sample in samples
    )

    return dict(
        sorted(
            label_counter.items()
        )
    )


def get_class_display_name(
    label: int,
) -> str:
    """
    정수 Label을 사람이 읽을 수 있는 클래스 이름으로 변환한다.

    Parameters
    ----------
    label:
        클래스 정수 Label이다.

    Returns
    -------
    str
        INDEX_TO_CLASS_NAME에 정의된 클래스 이름이다.

        Mapping에 없는 Label이면 LABEL_<숫자> 형식으로 반환한다.
    """

    return INDEX_TO_CLASS_NAME.get(
        label,
        f"LABEL_{label}",
    )


def print_sample_distribution(
    split_name: str,
    samples: Sequence[ImageSample],
) -> None:
    """
    하나의 Split에 대한 전체 수와 클래스 분포를 출력한다.

    Parameters
    ----------
    split_name:
        출력할 Split 이름이다.

        예:

            ORIGINAL TRAIN
            NEW TRAIN
            VALIDATION

    samples:
        클래스 분포를 계산할 이미지 Sample이다.
    """

    total_count = len(samples)

    label_counts = count_samples_by_label(
        samples=samples,
    )

    print(
        f"[{split_name}]"
    )

    print(
        f"total: {total_count}"
    )

    for label, count in label_counts.items():
        class_name = get_class_display_name(
            label=label,
        )

        ratio = (
            count
            / total_count
            * 100.0
        )

        print(
            f"{class_name:<10} "
            f"label={label} "
            f"count={count} "
            f"ratio={ratio:.2f}%"
        )

    print()


def print_split_summary(
    all_samples: Sequence[ImageSample],
    train_samples: Sequence[ImageSample],
    validation_samples: Sequence[ImageSample],
    validation_ratio: float,
    random_seed: int,
) -> None:
    """
    Train·Validation 분리 결과를 사람이 확인할 수 있도록 출력한다.

    Parameters
    ----------
    all_samples:
        기존 Train 디렉터리에서 수집한 전체 Sample이다.

    train_samples:
        새롭게 분리된 Train Sample이다.

    validation_samples:
        새롭게 분리된 Validation Sample이다.

    validation_ratio:
        분리에 사용한 Validation 비율이다.

    random_seed:
        분리에 사용한 Random Seed다.
    """

    print(
        "=" * 80
    )

    print(
        "DAY 2 - TRAIN / VALIDATION "
        "STRATIFIED SPLIT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "[CONFIGURATION]"
    )

    print(
        f"train root       : {TRAIN_ROOT}"
    )

    print(
        "validation ratio : "
        f"{validation_ratio:.2f}"
    )

    print(
        f"random seed      : {random_seed}"
    )

    print(
        "stratified split : True"
    )

    print()

    print_sample_distribution(
        split_name="ORIGINAL TRAIN",
        samples=all_samples,
    )

    print_sample_distribution(
        split_name="NEW TRAIN",
        samples=train_samples,
    )

    print_sample_distribution(
        split_name="VALIDATION",
        samples=validation_samples,
    )

    print(
        "[VALIDATION]"
    )

    print(
        "sample count preserved: "
        f"{len(all_samples) == len(train_samples) + len(validation_samples)}"
    )

    print(
        "train-validation overlap: False"
    )

    print()

    print(
        "[IMPORTANT]"
    )

    print(
        "The existing test split was not used "
        "during this operation."
    )

    print(
        "The existing test dataset remains "
        "reserved for final evaluation."
    )


def main() -> None:
    """
    실제 데이터셋을 사용하여 Train·Validation 분리를 실행한다.

    실행 명령
    ---------
    프로젝트 Root에서 다음 명령을 사용한다.

        python -m src.data.dataset_split

    실행 순서
    ---------
    TRAIN_ROOT

        → 이미지 Path·Label 수집

        → Stratified Train·Validation Split

        → 중복 검증

        → 클래스 분포 출력
    """

    all_samples = collect_image_samples(
        data_root=TRAIN_ROOT,
    )

    train_samples, validation_samples = (
        split_train_validation_samples(
            samples=all_samples,
            validation_ratio=(
                VALIDATION_RATIO
            ),
            random_seed=RANDOM_SEED,
        )
    )

    print_split_summary(
        all_samples=all_samples,
        train_samples=train_samples,
        validation_samples=(
            validation_samples
        ),
        validation_ratio=(
            VALIDATION_RATIO
        ),
        random_seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
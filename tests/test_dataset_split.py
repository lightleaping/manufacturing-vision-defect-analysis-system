"""
Train·Validation 이미지 Sample 분리 기능 테스트.

이 테스트 파일의 책임
---------------------
1. 이미지 확장자 정규화 기능을 검증한다.
2. 클래스 디렉터리에서 이미지 Path와 Label을 올바르게 수집하는지 검증한다.
3. 잘못된 데이터 구조에서 명확한 예외가 발생하는지 검증한다.
4. Stratified Train·Validation Split 결과를 검증한다.
5. Random Seed를 사용한 분리 재현성을 검증한다.
6. Train과 Validation 사이에 동일 이미지 Path가 없는지 검증한다.
7. 실제 Casting 데이터의 전체 수와 클래스별 분리 결과를 검증한다.

중요
----
현재 dataset_split.py는 실제 이미지 내용을 읽지 않는다.

따라서 임시 테스트 데이터에서는 빈 .jpeg 파일을 생성해도 된다.

실제 이미지의 정상 여부, RGB Mode, Width, Height 검증은 Day 1의
dataset_analysis 테스트가 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset_config import (
    CLASS_TO_INDEX,
    TRAIN_ROOT,
)
from src.data.dataset_split import (
    RANDOM_SEED,
    VALIDATION_RATIO,
    ImageSample,
    collect_image_samples,
    count_samples_by_label,
    normalize_image_extensions,
    split_train_validation_samples,
    validate_no_sample_overlap,
    validate_split_arguments,
)


def create_test_class_directories(
    data_root: Path,
) -> tuple[Path, Path]:
    """
    테스트용 NORMAL·DEFECT 클래스 디렉터리를 생성한다.

    Parameters
    ----------
    data_root:
        클래스 폴더를 생성할 임시 데이터 Root다.

    Returns
    -------
    tuple[Path, Path]
        첫 번째 Path:

            ok_front 디렉터리

        두 번째 Path:

            def_front 디렉터리

    필요한 이유
    -----------
    collect_image_samples()는 실제 프로젝트와 같은 디렉터리 구조를
    기대한다.

        data_root/
        ├── ok_front/
        └── def_front/

    여러 테스트에서 같은 폴더 생성 코드를 반복하지 않도록
    테스트 Helper 함수로 분리한다.
    """

    normal_directory = (
        data_root
        / "ok_front"
    )

    defect_directory = (
        data_root
        / "def_front"
    )

    normal_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    defect_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        normal_directory,
        defect_directory,
    )


def create_empty_image_file(
    image_path: Path,
) -> None:
    """
    이미지 경로 수집 테스트용 빈 파일을 생성한다.

    Parameters
    ----------
    image_path:
        생성할 테스트 파일 Path다.

    중요
    ----
    현재 collect_image_samples()는 이미지 내용을 열지 않고
    파일 Path와 확장자만 확인한다.

    따라서 이 테스트에서는 실제 JPEG Binary를 만들 필요가 없다.

    실제 이미지 유효성 검사는 Day 1의 dataset_analysis가 담당한다.
    """

    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path.touch()


def create_balanced_test_samples(
    sample_count_per_class: int = 10,
) -> list[ImageSample]:
    """
    Stratified Split 테스트용 가상 ImageSample 목록을 생성한다.

    Parameters
    ----------
    sample_count_per_class:
        클래스마다 생성할 Sample 수다.

    Returns
    -------
    list[ImageSample]
        NORMAL과 DEFECT가 같은 개수로 포함된 Sample 목록이다.

    예
    --
    sample_count_per_class=10이면:

        NORMAL 10개

        DEFECT 10개

        전체 20개

    실제 파일은 생성하지 않는다.

    split_train_validation_samples()는 ImageSample의 Path와 Label만
    사용하므로 분리 로직 테스트에는 가상 Path만으로 충분하다.
    """

    samples: list[ImageSample] = []

    for index in range(
        sample_count_per_class
    ):
        samples.append(
            ImageSample(
                image_path=Path(
                    f"normal_{index}.jpeg"
                ),
                label=0,
            )
        )

        samples.append(
            ImageSample(
                image_path=Path(
                    f"defect_{index}.jpeg"
                ),
                label=1,
            )
        )

    return samples


def test_normalize_image_extensions_adds_dot_and_lowercase() -> None:
    """
    확장자가 소문자이며 점으로 시작하는 형식으로 변환되는지 검증한다.
    """

    result = normalize_image_extensions(
        supported_extensions={
            "JPEG",
            ".JPG",
            " png ",
            "",
        },
    )

    assert result == {
        ".jpeg",
        ".jpg",
        ".png",
    }


def test_normalize_image_extensions_raises_for_empty_result() -> None:
    """
    유효한 이미지 확장자가 없으면 ValueError가 발생하는지 검증한다.
    """

    with pytest.raises(
        ValueError,
        match="지원 이미지 확장자가 비어 있습니다",
    ):
        normalize_image_extensions(
            supported_extensions={
                "",
                " ",
            },
        )


def test_collect_image_samples_connects_paths_and_labels(
    tmp_path: Path,
) -> None:
    """
    클래스 폴더의 이미지가 올바른 Label과 연결되는지 검증한다.
    """

    normal_directory, defect_directory = (
        create_test_class_directories(
            data_root=tmp_path,
        )
    )

    create_empty_image_file(
        normal_directory
        / "normal_1.jpeg"
    )

    create_empty_image_file(
        normal_directory
        / "normal_2.JPEG"
    )

    create_empty_image_file(
        defect_directory
        / "defect_1.jpeg"
    )

    # 지원하지 않는 확장자는 Sample에 포함되면 안 된다.
    (
        normal_directory
        / "readme.txt"
    ).write_text(
        "not an image",
        encoding="utf-8",
    )

    samples = collect_image_samples(
        data_root=tmp_path,
        class_to_index={
            "ok_front": 0,
            "def_front": 1,
        },
        supported_extensions={
            ".jpeg",
        },
    )

    assert len(samples) == 3

    label_counts = (
        count_samples_by_label(
            samples=samples,
        )
    )

    assert label_counts == {
        0: 2,
        1: 1,
    }

    sample_by_name = {
        sample.image_path.name: sample.label
        for sample in samples
    }

    assert sample_by_name == {
        "normal_1.jpeg": 0,
        "normal_2.JPEG": 0,
        "defect_1.jpeg": 1,
    }


def test_collect_image_samples_raises_for_missing_data_root(
    tmp_path: Path,
) -> None:
    """
    존재하지 않는 데이터 Root에서 FileNotFoundError가 발생하는지 검증한다.
    """

    missing_data_root = (
        tmp_path
        / "missing"
    )

    with pytest.raises(
        FileNotFoundError,
        match="데이터 Root를 찾을 수 없습니다",
    ):
        collect_image_samples(
            data_root=missing_data_root,
        )


def test_collect_image_samples_raises_for_missing_class_directory(
    tmp_path: Path,
) -> None:
    """
    필수 클래스 폴더가 누락되면 FileNotFoundError가 발생하는지 검증한다.
    """

    normal_directory = (
        tmp_path
        / "ok_front"
    )

    normal_directory.mkdir(
        parents=True,
    )

    create_empty_image_file(
        normal_directory
        / "normal.jpeg"
    )

    # def_front는 의도적으로 생성하지 않는다.
    with pytest.raises(
        FileNotFoundError,
        match="클래스 디렉터리를 찾을 수 없습니다",
    ):
        collect_image_samples(
            data_root=tmp_path,
        )


def test_collect_image_samples_raises_for_empty_class_directory(
    tmp_path: Path,
) -> None:
    """
    클래스 폴더는 있지만 지원 이미지가 없으면 ValueError가 발생하는지 검증한다.
    """

    normal_directory, _ = (
        create_test_class_directories(
            data_root=tmp_path,
        )
    )

    create_empty_image_file(
        normal_directory
        / "normal.jpeg"
    )

    # def_front 폴더에는 이미지 파일을 생성하지 않는다.
    with pytest.raises(
        ValueError,
        match="지원 이미지 파일이 없는 클래스입니다",
    ):
        collect_image_samples(
            data_root=tmp_path,
        )


@pytest.mark.parametrize(
    "validation_ratio",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_validate_split_arguments_rejects_invalid_ratio(
    validation_ratio: float,
) -> None:
    """
    Validation 비율이 유효 범위를 벗어나면 ValueError가 발생하는지 검증한다.
    """

    samples = (
        create_balanced_test_samples()
    )

    with pytest.raises(
        ValueError,
        match=(
            "validation_ratio는 0보다 크고 "
            "1보다 작아야 합니다"
        ),
    ):
        validate_split_arguments(
            samples=samples,
            validation_ratio=(
                validation_ratio
            ),
        )


def test_validate_split_arguments_requires_two_classes() -> None:
    """
    한 클래스만 존재하면 Stratified Split을 거부하는지 검증한다.
    """

    samples = [
        ImageSample(
            image_path=Path(
                f"normal_{index}.jpeg"
            ),
            label=0,
        )
        for index in range(10)
    ]

    with pytest.raises(
        ValueError,
        match="최소 두 개의 클래스가 필요합니다",
    ):
        validate_split_arguments(
            samples=samples,
            validation_ratio=0.20,
        )


def test_split_train_validation_samples_preserves_count_and_ratio() -> None:
    """
    전체 Sample 수와 클래스 비율이 분리 후에도 유지되는지 검증한다.
    """

    samples = (
        create_balanced_test_samples(
            sample_count_per_class=10,
        )
    )

    train_samples, validation_samples = (
        split_train_validation_samples(
            samples=samples,
            validation_ratio=0.20,
            random_seed=42,
        )
    )

    assert len(samples) == 20

    assert len(train_samples) == 16

    assert len(validation_samples) == 4

    assert (
        len(train_samples)
        + len(validation_samples)
        == len(samples)
    )

    assert count_samples_by_label(
        samples=train_samples,
    ) == {
        0: 8,
        1: 8,
    }

    assert count_samples_by_label(
        samples=validation_samples,
    ) == {
        0: 2,
        1: 2,
    }


def test_split_train_validation_samples_is_reproducible() -> None:
    """
    같은 Seed를 사용하면 동일한 이미지가 동일한 Split에 배정되는지 검증한다.
    """

    samples = (
        create_balanced_test_samples(
            sample_count_per_class=20,
        )
    )

    first_train, first_validation = (
        split_train_validation_samples(
            samples=samples,
            validation_ratio=0.20,
            random_seed=RANDOM_SEED,
        )
    )

    second_train, second_validation = (
        split_train_validation_samples(
            samples=samples,
            validation_ratio=0.20,
            random_seed=RANDOM_SEED,
        )
    )

    first_train_paths = [
        sample.image_path
        for sample in first_train
    ]

    second_train_paths = [
        sample.image_path
        for sample in second_train
    ]

    first_validation_paths = [
        sample.image_path
        for sample in first_validation
    ]

    second_validation_paths = [
        sample.image_path
        for sample in second_validation
    ]

    assert (
        first_train_paths
        == second_train_paths
    )

    assert (
        first_validation_paths
        == second_validation_paths
    )


def test_validate_no_sample_overlap_accepts_separate_paths() -> None:
    """
    Train과 Validation Path가 서로 다르면 예외가 발생하지 않는지 검증한다.
    """

    train_samples = [
        ImageSample(
            image_path=Path(
                "train_normal.jpeg"
            ),
            label=0,
        ),
    ]

    validation_samples = [
        ImageSample(
            image_path=Path(
                "validation_normal.jpeg"
            ),
            label=0,
        ),
    ]

    validate_no_sample_overlap(
        train_samples=train_samples,
        validation_samples=(
            validation_samples
        ),
    )


def test_validate_no_sample_overlap_raises_for_duplicate_path() -> None:
    """
    동일 이미지 Path가 두 Split에 있으면 ValueError가 발생하는지 검증한다.
    """

    duplicated_path = Path(
        "same_image.jpeg"
    )

    train_samples = [
        ImageSample(
            image_path=duplicated_path,
            label=0,
        ),
    ]

    validation_samples = [
        ImageSample(
            image_path=duplicated_path,
            label=0,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Train과 Validation에 "
            "중복 이미지가 있습니다"
        ),
    ):
        validate_no_sample_overlap(
            train_samples=train_samples,
            validation_samples=(
                validation_samples
            ),
        )


def test_real_train_dataset_sample_count_and_labels() -> None:
    """
    실제 Casting Train 데이터에서 전체 수와 클래스별 수를 검증한다.

    Day 1 분석 결과
    ----------------
    NORMAL:

        2,875장

    DEFECT:

        3,758장

    전체:

        6,633장
    """

    samples = collect_image_samples(
        data_root=TRAIN_ROOT,
    )

    assert len(samples) == 6_633

    assert count_samples_by_label(
        samples=samples,
    ) == {
        CLASS_TO_INDEX["ok_front"]: 2_875,
        CLASS_TO_INDEX["def_front"]: 3_758,
    }


def test_real_train_validation_split_counts() -> None:
    """
    실제 Casting 데이터의 Stratified Split 결과를 검증한다.

    현재 고정 설정
    --------------
    validation_ratio:

        0.20

    random_seed:

        42

    기대 결과
    ---------
    Train:

        전체 5,306장

        NORMAL 2,300장

        DEFECT 3,006장

    Validation:

        전체 1,327장

        NORMAL 575장

        DEFECT 752장
    """

    samples = collect_image_samples(
        data_root=TRAIN_ROOT,
    )

    train_samples, validation_samples = (
        split_train_validation_samples(
            samples=samples,
            validation_ratio=(
                VALIDATION_RATIO
            ),
            random_seed=RANDOM_SEED,
        )
    )

    assert len(train_samples) == 5_306

    assert len(validation_samples) == 1_327

    assert count_samples_by_label(
        samples=train_samples,
    ) == {
        CLASS_TO_INDEX["ok_front"]: 2_300,
        CLASS_TO_INDEX["def_front"]: 3_006,
    }

    assert count_samples_by_label(
        samples=validation_samples,
    ) == {
        CLASS_TO_INDEX["ok_front"]: 575,
        CLASS_TO_INDEX["def_front"]: 752,
    }

    train_paths = {
        sample.image_path.resolve()
        for sample in train_samples
    }

    validation_paths = {
        sample.image_path.resolve()
        for sample in validation_samples
    }

    assert (
        train_paths
        & validation_paths
    ) == set()

    assert (
        len(train_samples)
        + len(validation_samples)
    ) == len(samples)
"""
PyTorch Vision DataLoader 테스트.

이 테스트 파일의 책임
---------------------
1. DataLoader 기본 설정값을 검증한다.
2. 잘못된 DataLoader 입력값의 예외 처리를 검증한다.
3. Train과 평가 DataLoader의 Sampler 구성을 검증한다.
4. Batch Size와 마지막 작은 Batch 처리 방식을 검증한다.
5. 같은 Random Seed에서 Shuffle 순서가 재현되는지 검증한다.
6. Image·Label Batch의 Shape와 dtype을 검증한다.
7. NaN·inf·잘못된 Label을 검증한다.
8. 실제 Casting Dataset·DataLoader 수를 검증한다.
9. 실제 Train·Validation·Test 첫 Batch를 검증한다.

테스트 구분
----------
단위 테스트:

    작은 TensorDataset을 사용한다.

    실제 이미지 전체를 반복해서 읽지 않고
    DataLoader 설정과 Batch 동작만 빠르게 검증한다.

통합 테스트:

    실제 Casting 이미지 데이터를 사용한다.

    Dataset

    → Transform

    → DataLoader

    → Batch

    전체 연결을 검증한다.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from src.data.data_loader import (
    BATCH_SIZE,
    DROP_LAST,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    VisionDataLoaders,
    create_data_loader,
    create_vision_data_loaders,
    validate_data_loader_arguments,
    validate_image_label_batch,
    validate_sampler_types,
)


def create_small_tensor_dataset(
    sample_count: int = 10,
) -> TensorDataset:
    """
    DataLoader 단위 테스트용 작은 TensorDataset을 생성한다.

    Parameters
    ----------
    sample_count:
        생성할 Sample 수다.

        기본값:

            10

    Returns
    -------
    TensorDataset
        Feature와 Label을 가진 작은 Dataset이다.

    Feature
    -------
    다음 숫자를 사용한다.

        0

        1

        2

        ...

    Shuffle 순서를 확인할 때 각 Sample을 구분하기 위해
    서로 다른 숫자를 사용한다.

    Label
    -----
    0과 1을 번갈아 생성한다.

        0

        1

        0

        1

        ...
    """

    features = torch.arange(
        sample_count,
        dtype=torch.float32,
    ).reshape(
        sample_count,
        1,
    )

    labels = (
        torch.arange(
            sample_count,
            dtype=torch.int64,
        )
        % 2
    )

    return TensorDataset(
        features,
        labels,
    )


def collect_feature_order(
    data_loader: DataLoader,
) -> Tensor:
    """
    DataLoader가 반환하는 Feature 순서를 하나의 Tensor로 합친다.

    Parameters
    ----------
    data_loader:
        순서를 확인할 DataLoader다.

    Returns
    -------
    Tensor
        DataLoader 반복 과정에서 반환된 전체 Feature 순서다.

    사용 목적
    ---------
    같은 Random Seed로 생성한 두 DataLoader가
    같은 Shuffle 순서를 만드는지 확인한다.
    """

    feature_batches: list[
        Tensor
    ] = []

    for features, _ in data_loader:
        feature_batches.append(
            features.reshape(-1)
        )

    return torch.cat(
        feature_batches
    )


@pytest.fixture(
    scope="module",
)
def real_data_loaders() -> (
    Iterator[VisionDataLoaders]
):
    """
    실제 Casting Dataset·DataLoader를 테스트 모듈에서 한 번 생성한다.

    scope="module"
    --------------
    실제 데이터에는 수천 개의 이미지 Path가 있다.

    각 테스트마다 전체 Dataset을 다시 생성하면
    같은 Path 검증을 불필요하게 반복한다.

    따라서 현재 테스트 파일 전체에서 한 번 생성한 객체를 공유한다.

    중요
    ----
    테스트 사이에서 Dataset 내용을 수정하지 않는다.
    """

    data_loaders = (
        create_vision_data_loaders()
    )

    yield data_loaders


def test_data_loader_constants_are_expected() -> None:
    """
    현재 프로젝트의 기본 DataLoader 설정을 검증한다.
    """

    assert BATCH_SIZE == 32

    assert NUM_WORKERS == 0

    assert PIN_MEMORY is False

    assert DROP_LAST is False

    assert (
        PERSISTENT_WORKERS
        is False
    )


def test_validate_data_loader_arguments_rejects_non_dataset() -> None:
    """
    PyTorch Dataset이 아닌 객체를 거부하는지 검증한다.
    """

    with pytest.raises(
        TypeError,
        match=(
            "dataset은 "
            "torch.utils.data.Dataset이어야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=object(),  # type: ignore[arg-type]
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )


def test_validate_data_loader_arguments_rejects_empty_dataset() -> None:
    """
    Sample이 없는 Dataset을 거부하는지 검증한다.
    """

    empty_dataset = TensorDataset(
        torch.empty(
            (
                0,
                1,
            ),
            dtype=torch.float32,
        ),
        torch.empty(
            (
                0,
            ),
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "DataLoader에 전달할 "
            "Dataset이 비어 있습니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=empty_dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )


@pytest.mark.parametrize(
    "invalid_batch_size",
    [
        True,
        4.0,
        "4",
    ],
)
def test_validate_data_loader_arguments_rejects_non_integer_batch_size(
    invalid_batch_size: object,
) -> None:
    """
    실제 int가 아닌 Batch Size를 거부하는지 검증한다.

    bool은 Python에서 int의 하위 타입이지만
    Batch Size로 허용하지 않는다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        TypeError,
        match=(
            "batch_size는 "
            "int여야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=dataset,
            batch_size=invalid_batch_size,  # type: ignore[arg-type]
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )


@pytest.mark.parametrize(
    "invalid_batch_size",
    [
        0,
        -1,
    ],
)
def test_validate_data_loader_arguments_rejects_non_positive_batch_size(
    invalid_batch_size: int,
) -> None:
    """
    0 이하의 Batch Size를 거부하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        ValueError,
        match=(
            "batch_size는 "
            "1 이상이어야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=dataset,
            batch_size=invalid_batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )


@pytest.mark.parametrize(
    "invalid_num_workers",
    [
        True,
        1.5,
        "0",
    ],
)
def test_validate_data_loader_arguments_rejects_non_integer_num_workers(
    invalid_num_workers: object,
) -> None:
    """
    실제 int가 아닌 num_workers를 거부하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        TypeError,
        match=(
            "num_workers는 "
            "int여야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=dataset,
            batch_size=4,
            num_workers=invalid_num_workers,  # type: ignore[arg-type]
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )


def test_validate_data_loader_arguments_rejects_negative_num_workers() -> None:
    """
    음수 Worker 수를 거부하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        ValueError,
        match=(
            "num_workers는 "
            "0 이상이어야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=dataset,
            batch_size=4,
            num_workers=-1,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )


@pytest.mark.parametrize(
    (
        "argument_name",
        "invalid_value",
    ),
    [
        (
            "pin_memory",
            1,
        ),
        (
            "drop_last",
            0,
        ),
        (
            "persistent_workers",
            "False",
        ),
    ],
)
def test_validate_data_loader_arguments_rejects_non_boolean_options(
    argument_name: str,
    invalid_value: object,
) -> None:
    """
    Boolean 설정에 bool이 아닌 값이 들어오면 거부하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    arguments: dict[
        str,
        object,
    ] = {
        "dataset": dataset,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
        "persistent_workers": False,
    }

    arguments[
        argument_name
    ] = invalid_value

    with pytest.raises(
        TypeError,
        match=(
            f"{argument_name}는 "
            "bool이어야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            **arguments,  # type: ignore[arg-type]
        )


def test_validate_data_loader_arguments_rejects_persistent_workers_without_workers() -> None:
    """
    num_workers=0에서 persistent_workers=True를 거부하는지 검증한다.

    유지할 별도 Worker Process가 없으므로
    두 설정을 동시에 사용할 수 없다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        ValueError,
        match=(
            "persistent_workers=True를 사용하려면 "
            "num_workers가 1 이상이어야 합니다"
        ),
    ):
        validate_data_loader_arguments(
            dataset=dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
        )


def test_create_data_loader_rejects_non_boolean_shuffle() -> None:
    """
    shuffle이 bool이 아니면 TypeError가 발생하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        TypeError,
        match=(
            "shuffle은 "
            "bool이어야 합니다"
        ),
    ):
        create_data_loader(
            dataset=dataset,
            shuffle=1,  # type: ignore[arg-type]
        )


def test_create_data_loader_rejects_non_integer_random_seed() -> None:
    """
    Random Seed가 실제 int가 아니면 거부하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    with pytest.raises(
        TypeError,
        match=(
            "random_seed는 "
            "int여야 합니다"
        ),
    ):
        create_data_loader(
            dataset=dataset,
            shuffle=True,
            random_seed=True,  # type: ignore[arg-type]
        )


def test_create_data_loader_uses_random_sampler_when_shuffle_is_true() -> None:
    """
    shuffle=True이면 RandomSampler를 사용하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    data_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
        )
    )

    assert isinstance(
        data_loader.sampler,
        RandomSampler,
    )


def test_create_data_loader_uses_sequential_sampler_when_shuffle_is_false() -> None:
    """
    shuffle=False이면 SequentialSampler를 사용하는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    data_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
        )
    )

    assert isinstance(
        data_loader.sampler,
        SequentialSampler,
    )


def test_create_data_loader_stores_expected_configuration() -> None:
    """
    DataLoader에 전달한 설정값이 실제 객체에 저장되는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset()
    )

    data_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            random_seed=123,
        )
    )

    assert (
        data_loader.batch_size
        == 4
    )

    assert (
        data_loader.num_workers
        == 0
    )

    assert (
        data_loader.pin_memory
        is False
    )

    assert (
        data_loader.drop_last
        is False
    )

    assert (
        data_loader.persistent_workers
        is False
    )

    assert (
        data_loader.generator
        is not None
    )

    assert (
        data_loader.generator.initial_seed()
        == 123
    )


def test_create_data_loader_keeps_last_incomplete_batch() -> None:
    """
    drop_last=False이면 마지막 작은 Batch를 유지하는지 검증한다.

    Dataset:

        10개

    Batch Size:

        4

    기대:

        4

        4

        2

    총 3 Batch
    """

    dataset = (
        create_small_tensor_dataset(
            sample_count=10,
        )
    )

    data_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            drop_last=False,
        )
    )

    batches = list(
        data_loader
    )

    assert len(
        batches
    ) == 3

    last_features, last_labels = (
        batches[-1]
    )

    assert (
        last_features.shape[0]
        == 2
    )

    assert (
        last_labels.shape[0]
        == 2
    )


def test_create_data_loader_drops_last_incomplete_batch() -> None:
    """
    drop_last=True이면 마지막 작은 Batch를 제거하는지 검증한다.

    Dataset:

        10개

    Batch Size:

        4

    기대:

        4

        4

    마지막 2개 Sample은 제외된다.
    """

    dataset = (
        create_small_tensor_dataset(
            sample_count=10,
        )
    )

    data_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            drop_last=True,
        )
    )

    batches = list(
        data_loader
    )

    assert len(
        batches
    ) == 2

    assert all(
        features.shape[0] == 4
        and labels.shape[0] == 4
        for features, labels
        in batches
    )


def test_create_data_loader_shuffle_is_reproducible_with_same_seed() -> None:
    """
    같은 Seed의 두 DataLoader가 같은 Shuffle 순서를 만드는지 검증한다.
    """

    dataset = (
        create_small_tensor_dataset(
            sample_count=20,
        )
    )

    first_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=5,
            shuffle=True,
            random_seed=42,
        )
    )

    second_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=5,
            shuffle=True,
            random_seed=42,
        )
    )

    first_order = (
        collect_feature_order(
            data_loader=first_loader,
        )
    )

    second_order = (
        collect_feature_order(
            data_loader=second_loader,
        )
    )

    assert torch.equal(
        first_order,
        second_order,
    )


def test_validate_image_label_batch_accepts_valid_batch() -> None:
    """
    정상 Image·Label Batch가 검증을 통과하는지 확인한다.
    """

    images = torch.randn(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    validate_image_label_batch(
        images=images,
        labels=labels,
    )


def test_validate_image_label_batch_rejects_non_tensor_images() -> None:
    """
    images가 Tensor가 아니면 TypeError가 발생하는지 검증한다.
    """

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        TypeError,
        match=(
            "images는 "
            "torch.Tensor여야 합니다"
        ),
    ):
        validate_image_label_batch(
            images=[],  # type: ignore[arg-type]
            labels=labels,
        )


def test_validate_image_label_batch_rejects_non_tensor_labels() -> None:
    """
    labels가 Tensor가 아니면 TypeError가 발생하는지 검증한다.
    """

    images = torch.randn(
        (
            2,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    with pytest.raises(
        TypeError,
        match=(
            "labels는 "
            "torch.Tensor여야 합니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=[0, 1],  # type: ignore[arg-type]
        )


def test_validate_image_label_batch_rejects_invalid_image_rank() -> None:
    """
    Batch 차원이 없는 3차원 Image Tensor를 거부하는지 검증한다.
    """

    images = torch.randn(
        (
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Image Batch는 4차원"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_empty_image_batch() -> None:
    """
    Batch Size가 0인 Image Tensor를 거부하는지 검증한다.
    """

    images = torch.empty(
        (
            0,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.empty(
        (
            0,
        ),
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Image Batch Size는 "
            "1 이상이어야 합니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (
            4,
            1,
            224,
            224,
        ),
        (
            4,
            3,
            128,
            128,
        ),
    ],
)
def test_validate_image_label_batch_rejects_invalid_image_shape(
    invalid_shape: tuple[
        int,
        int,
        int,
        int,
    ],
) -> None:
    """
    RGB Channel 또는 공간 크기가 잘못된 Batch를 거부하는지 검증한다.
    """

    images = torch.randn(
        invalid_shape,
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "개별 이미지 Shape는 "
            r"\[3, 224, 224\]여야 합니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_non_floating_images() -> None:
    """
    정수 Image Batch를 거부하는지 검증한다.
    """

    images = torch.zeros(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.uint8,
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Image Batch는 "
            "Floating Point"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_non_finite_images() -> None:
    """
    NaN이 포함된 Image Batch를 거부하는지 검증한다.
    """

    images = torch.zeros(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    images[
        0,
        0,
        0,
        0,
    ] = float(
        "nan"
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Image Batch에 "
            "NaN 또는 inf가 있습니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_invalid_label_rank() -> None:
    """
    [B, 1] 형태의 2차원 Label Batch를 거부하는지 검증한다.
    """

    images = torch.randn(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            [
                0,
            ],
            [
                1,
            ],
            [
                0,
            ],
            [
                1,
            ],
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Label Batch는 1차원"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_invalid_label_dtype() -> None:
    """
    torch.int64가 아닌 Label Batch를 거부하는지 검증한다.
    """

    images = torch.randn(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int32,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Label Batch dtype은 "
            "torch.int64여야 합니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_mismatched_sample_counts() -> None:
    """
    이미지 수와 Label 수가 다르면 ValueError가 발생하는지 검증한다.
    """

    images = torch.randn(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Image Batch와 Label Batch의 "
            "Sample 수가 다릅니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_validate_image_label_batch_rejects_unknown_label() -> None:
    """
    현재 프로젝트 Label인 0·1 이외의 값을 거부하는지 검증한다.
    """

    images = torch.randn(
        (
            4,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            2,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "0 또는 1만 "
            "포함되어야 합니다"
        ),
    ):
        validate_image_label_batch(
            images=images,
            labels=labels,
        )


def test_real_vision_dataset_counts(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    실제 Casting Train·Validation·Test Dataset 수를 검증한다.
    """

    assert len(
        real_data_loaders.train_dataset
    ) == 5_306

    assert len(
        real_data_loaders.validation_dataset
    ) == 1_327

    assert len(
        real_data_loaders.test_dataset
    ) == 715


def test_real_vision_data_loader_batch_counts(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    실제 Dataset의 Batch 수를 검증한다.

    drop_last=False이므로 마지막 작은 Batch도 포함한다.
    """

    assert len(
        real_data_loaders.train_loader
    ) == 166

    assert len(
        real_data_loaders.validation_loader
    ) == 42

    assert len(
        real_data_loaders.test_loader
    ) == 23


def test_real_vision_data_loader_configuration(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    실제 세 DataLoader가 현재 CPU·Windows 기본 설정을 사용하는지 검증한다.
    """

    data_loader_list = [
        real_data_loaders.train_loader,
        real_data_loaders.validation_loader,
        real_data_loaders.test_loader,
    ]

    for data_loader in (
        data_loader_list
    ):
        assert (
            data_loader.batch_size
            == 32
        )

        assert (
            data_loader.num_workers
            == 0
        )

        assert (
            data_loader.pin_memory
            is False
        )

        assert (
            data_loader.drop_last
            is False
        )

        assert (
            data_loader.persistent_workers
            is False
        )


def test_real_vision_data_loader_sampler_types(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    실제 Train은 RandomSampler,
    Validation·Test는 SequentialSampler인지 검증한다.
    """

    assert isinstance(
        real_data_loaders
        .train_loader
        .sampler,
        RandomSampler,
    )

    assert isinstance(
        real_data_loaders
        .validation_loader
        .sampler,
        SequentialSampler,
    )

    assert isinstance(
        real_data_loaders
        .test_loader
        .sampler,
        SequentialSampler,
    )

    # 프로젝트 검증 함수도 같은 구성을 정상 처리해야 한다.
    validate_sampler_types(
        data_loaders=(
            real_data_loaders
        ),
    )


def test_real_first_batches_are_valid(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    실제 Train·Validation·Test 첫 Batch가
    모델 입력 조건을 만족하는지 검증한다.
    """

    data_loader_list = [
        real_data_loaders.train_loader,
        real_data_loaders.validation_loader,
        real_data_loaders.test_loader,
    ]

    for data_loader in (
        data_loader_list
    ):
        images, labels = next(
            iter(
                data_loader
            )
        )

        validate_image_label_batch(
            images=images,
            labels=labels,
        )

        assert images.shape == (
            32,
            3,
            224,
            224,
        )

        assert labels.shape == (
            32,
        )

        assert (
            images.dtype
            == torch.float32
        )

        assert (
            labels.dtype
            == torch.int64
        )


def test_real_evaluation_first_batches_are_deterministic(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    Validation·Test는 SequentialSampler와 고정 Transform을 사용하므로
    새 Iterator를 만들 때 같은 첫 Batch가 생성되는지 검증한다.
    """

    evaluation_loaders = [
        (
            real_data_loaders
            .validation_loader
        ),
        (
            real_data_loaders
            .test_loader
        ),
    ]

    for data_loader in (
        evaluation_loaders
    ):
        (
            first_images,
            first_labels,
        ) = next(
            iter(
                data_loader
            )
        )

        (
            second_images,
            second_labels,
        ) = next(
            iter(
                data_loader
            )
        )

        assert torch.equal(
            first_images,
            second_images,
        )

        assert torch.equal(
            first_labels,
            second_labels,
        )


def test_validate_sampler_types_rejects_sequential_train_loader(
    real_data_loaders: VisionDataLoaders,
) -> None:
    """
    Train DataLoader가 잘못 SequentialSampler를 사용하면
    프로젝트 검증 함수가 오류를 발생시키는지 확인한다.
    """

    wrong_train_loader = (
        create_data_loader(
            dataset=(
                real_data_loaders
                .train_dataset
            ),
            batch_size=32,
            shuffle=False,
        )
    )

    invalid_data_loaders = (
        VisionDataLoaders(
            train_dataset=(
                real_data_loaders
                .train_dataset
            ),
            validation_dataset=(
                real_data_loaders
                .validation_dataset
            ),
            test_dataset=(
                real_data_loaders
                .test_dataset
            ),
            train_loader=(
                wrong_train_loader
            ),
            validation_loader=(
                real_data_loaders
                .validation_loader
            ),
            test_loader=(
                real_data_loaders
                .test_loader
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Train DataLoader는 "
            "RandomSampler를 사용해야 합니다"
        ),
    ):
        validate_sampler_types(
            data_loaders=(
                invalid_data_loaders
            ),
        )
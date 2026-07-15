"""
PyTorch Dataset·DataLoader 통합 구성 모듈.

이 모듈의 책임
----------------
1. 기존 Train 이미지 Path와 Label을 수집한다.
2. 기존 Train을 새로운 Train·Validation으로 분리한다.
3. 기존 Test 이미지는 최종 평가용으로 별도 수집한다.
4. Train·Validation·Test Dataset을 생성한다.
5. 각 Dataset에 역할에 맞는 Transform을 연결한다.
6. Train·Validation·Test DataLoader를 생성한다.
7. Batch Size·Shuffle·Worker·Pinned Memory 설정을 관리한다.
8. Dataset 수·Batch 수·첫 Batch Tensor 정보를 출력한다.

전체 데이터 흐름
---------------
casting_data/train

→ ImageSample 수집

→ Stratified Split

├── Train Sample
│
│   → Train Transform
│
│   → CastingDefectDataset
│
│   → Train DataLoader
│
└── Validation Sample

    → Validation Transform

    → CastingDefectDataset

    → Validation DataLoader


casting_data/test

→ Test Sample 수집

→ Test Transform

→ CastingDefectDataset

→ Test DataLoader

중요
----
이 모듈은 모델을 생성하거나 학습하지 않는다.

모델 정의·Loss·Optimizer·학습 Loop는 이후 Day에서 별도로 구현한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from src.data.dataset_config import (
    TEST_ROOT,
    TRAIN_ROOT,
)
from src.data.dataset_split import (
    RANDOM_SEED,
    VALIDATION_RATIO,
    collect_image_samples,
    split_train_validation_samples,
)
from src.data.image_dataset import (
    CastingDefectDataset,
)
from src.data.image_transforms import (
    create_test_transform,
    create_train_transform,
    create_validation_transform,
)


# 한 번의 모델 Forward에 전달할 이미지 수다.
#
# 현재 기본값:
#
#     32장
#
# 향후 실제 CNN·ResNet18 학습 속도와 메모리 사용량을 확인한 뒤
# 필요하면 16 또는 다른 값과 비교할 수 있다.
BATCH_SIZE: int = 32


# DataLoader에서 이미지 로드와 Transform에 사용할
# 별도 Worker Process 수다.
#
# 현재 환경:
#
#     Windows
#
#     CPU 전용
#
#     Intel Core i5-1035G7
#
# Day 2에서는 안정적인 실행과 오류 추적을 우선하여
# Main Process만 사용하는 0으로 시작한다.
NUM_WORKERS: int = 0


# CPU Memory를 Page-Locked Memory로 고정할지 결정한다.
#
# 현재는 CUDA GPU를 사용하지 않으므로 False다.
#
# 향후 NVIDIA CUDA 환경에서는 실제 전송 성능을 측정한 뒤
# True 사용을 검토할 수 있다.
PIN_MEMORY: bool = False


# Dataset 전체 크기가 Batch Size로 나누어떨어지지 않을 때
# 마지막 작은 Batch를 유지할지 결정한다.
#
# False:
#
#     마지막 작은 Batch도 사용
#
# 현재는 모든 Train·Validation·Test 이미지를 사용하기 위해
# False로 설정한다.
DROP_LAST: bool = False


# Worker Process를 Epoch 사이에 계속 유지할지 결정한다.
#
# 현재 NUM_WORKERS=0이므로 별도 Worker Process가 없다.
#
# 따라서 False를 사용한다.
PERSISTENT_WORKERS: bool = False


@dataclass(frozen=True)
class VisionDataLoaders:
    """
    Train·Validation·Test Dataset과 DataLoader를 하나로 보관한다.

    Attributes
    ----------
    train_dataset:
        Train Sample과 Train Transform을 사용하는 Dataset이다.

    validation_dataset:
        Validation Sample과 고정 평가 Transform을 사용하는 Dataset이다.

    test_dataset:
        기존 Test Sample과 고정 평가 Transform을 사용하는 Dataset이다.

    train_loader:
        Train Dataset을 Batch로 공급하는 DataLoader다.

        설정:

            shuffle=True

    validation_loader:
        Validation Dataset을 Batch로 공급하는 DataLoader다.

        설정:

            shuffle=False

    test_loader:
        Test Dataset을 Batch로 공급하는 DataLoader다.

        설정:

            shuffle=False

    frozen=True
    -----------
    생성 이후 Dataset·DataLoader 참조가 실수로 다른 객체로
    변경되지 않도록 불변 데이터 구조로 관리한다.
    """

    train_dataset: CastingDefectDataset

    validation_dataset: CastingDefectDataset

    test_dataset: CastingDefectDataset

    train_loader: DataLoader

    validation_loader: DataLoader

    test_loader: DataLoader


def validate_data_loader_arguments(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool,
) -> None:
    """
    DataLoader 생성 전에 주요 설정값을 검증한다.

    Parameters
    ----------
    dataset:
        DataLoader가 Batch로 묶을 PyTorch Dataset이다.

    batch_size:
        한 Batch에 포함할 Sample 수다.

    num_workers:
        데이터 로드에 사용할 Worker Process 수다.

    pin_memory:
        Pinned Memory 사용 여부다.

    drop_last:
        마지막 작은 Batch 제거 여부다.

    persistent_workers:
        Worker Process 유지 여부다.

    Raises
    ------
    TypeError
        Dataset 또는 Boolean 설정값의 타입이 잘못되었을 때 발생한다.

    ValueError
        Dataset이 비어 있거나 정수 설정값이 유효하지 않을 때 발생한다.

        num_workers=0인데 persistent_workers=True인 경우 발생한다.

    필요한 이유
    -----------
    잘못된 값을 DataLoader 내부까지 전달하기 전에
    프로젝트 수준에서 더 명확한 오류 메시지를 제공한다.
    """

    if not isinstance(
        dataset,
        Dataset,
    ):
        raise TypeError(
            "dataset은 torch.utils.data.Dataset이어야 합니다: "
            f"{type(dataset).__name__}"
        )

    if len(dataset) <= 0:
        raise ValueError(
            "DataLoader에 전달할 Dataset이 비어 있습니다."
        )

    # bool은 Python에서 int의 하위 타입이다.
    #
    # True를 batch_size=1로 잘못 허용하지 않기 위해
    # type(...) is int를 사용한다.
    if type(batch_size) is not int:
        raise TypeError(
            "batch_size는 int여야 합니다: "
            f"{type(batch_size).__name__}"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size는 1 이상이어야 합니다: "
            f"{batch_size}"
        )

    if type(num_workers) is not int:
        raise TypeError(
            "num_workers는 int여야 합니다: "
            f"{type(num_workers).__name__}"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers는 0 이상이어야 합니다: "
            f"{num_workers}"
        )

    if type(pin_memory) is not bool:
        raise TypeError(
            "pin_memory는 bool이어야 합니다: "
            f"{type(pin_memory).__name__}"
        )

    if type(drop_last) is not bool:
        raise TypeError(
            "drop_last는 bool이어야 합니다: "
            f"{type(drop_last).__name__}"
        )

    if type(persistent_workers) is not bool:
        raise TypeError(
            "persistent_workers는 bool이어야 합니다: "
            f"{type(persistent_workers).__name__}"
        )

    # 유지할 Worker Process가 하나도 없는데
    # persistent_workers=True이면 설정 의미가 없다.
    if (
        persistent_workers
        and num_workers == 0
    ):
        raise ValueError(
            "persistent_workers=True를 사용하려면 "
            "num_workers가 1 이상이어야 합니다."
        )


def create_data_loader(
    dataset: Dataset,
    *,
    batch_size: int = BATCH_SIZE,
    shuffle: bool,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    drop_last: bool = DROP_LAST,
    persistent_workers: bool = (
        PERSISTENT_WORKERS
    ),
    random_seed: int = RANDOM_SEED,
) -> DataLoader:
    """
    하나의 Dataset에 대한 PyTorch DataLoader를 생성한다.

    Parameters
    ----------
    dataset:
        Batch로 묶을 PyTorch Dataset이다.

    batch_size:
        한 Batch에 포함할 최대 Sample 수다.

        기본값:

            32

    shuffle:
        Epoch마다 Sample 순서를 섞을지 결정한다.

        Train:

            True

        Validation·Test:

            False

    num_workers:
        데이터 로드 Worker Process 수다.

        현재 기본값:

            0

    pin_memory:
        Pinned Memory 사용 여부다.

        현재 CPU 전용 환경:

            False

    drop_last:
        마지막 작은 Batch를 버릴지 결정한다.

        현재:

            False

        모든 이미지를 사용한다.

    persistent_workers:
        Worker Process를 DataLoader 반복 이후에도 유지할지 결정한다.

        현재 num_workers=0:

            False

    random_seed:
        DataLoader의 랜덤 순서를 재현하기 위한 Seed다.

        기본값:

            42

    Returns
    -------
    torch.utils.data.DataLoader
        설정이 적용된 DataLoader다.

    Random Generator
    ----------------
    DataLoader마다 독립적인 torch.Generator를 생성한다.

    같은 Generator 객체를 Train·Validation·Test가 공유하면
    한 DataLoader의 반복이 다른 DataLoader의 Random State에
    영향을 줄 수 있다.

    따라서 각 DataLoader 생성 시 새로운 Generator를 사용한다.
    """

    if type(shuffle) is not bool:
        raise TypeError(
            "shuffle은 bool이어야 합니다: "
            f"{type(shuffle).__name__}"
        )

    if type(random_seed) is not int:
        raise TypeError(
            "random_seed는 int여야 합니다: "
            f"{type(random_seed).__name__}"
        )

    validate_data_loader_arguments(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(
            persistent_workers
        ),
    )

    # Shuffle 순서 재현을 위한 독립 Generator다.
    data_loader_generator = (
        torch.Generator()
    )

    data_loader_generator.manual_seed(
        random_seed
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(
            persistent_workers
        ),
        generator=data_loader_generator,
    )


def create_vision_data_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    drop_last: bool = DROP_LAST,
    persistent_workers: bool = (
        PERSISTENT_WORKERS
    ),
    random_seed: int = RANDOM_SEED,
) -> VisionDataLoaders:
    """
    Train·Validation·Test Dataset과 DataLoader 전체를 생성한다.

    Parameters
    ----------
    batch_size:
        Train·Validation·Test에 공통 적용할 Batch Size다.

    num_workers:
        DataLoader Worker Process 수다.

    pin_memory:
        Pinned Memory 사용 여부다.

    drop_last:
        마지막 작은 Batch 제거 여부다.

    persistent_workers:
        Worker Process 유지 여부다.

    random_seed:
        Train·Validation Split과 DataLoader Shuffle에 사용할 Seed다.

    Returns
    -------
    VisionDataLoaders
        Train·Validation·Test Dataset과 DataLoader를 보관한 객체다.

    전체 호출 관계
    -------------
    TRAIN_ROOT

    → collect_image_samples()

    → split_train_validation_samples()

    → Train Sample

    → Validation Sample


    TEST_ROOT

    → collect_image_samples()

    → Test Sample


    각 Sample

    → CastingDefectDataset

    → 역할별 Transform

    → create_data_loader()

    → VisionDataLoaders
    """

    # 기존 Train 6,633장의 이미지 Path와 Label을 수집한다.
    original_train_samples = (
        collect_image_samples(
            data_root=TRAIN_ROOT,
        )
    )

    # 기존 Train만 새로운 Train·Validation으로 분리한다.
    #
    # 기존 Test는 이 분리 과정에 사용하지 않는다.
    (
        train_samples,
        validation_samples,
    ) = split_train_validation_samples(
        samples=original_train_samples,
        validation_ratio=(
            VALIDATION_RATIO
        ),
        random_seed=random_seed,
    )

    # 기존 Test 715장은 최종 평가용으로 별도 수집한다.
    test_samples = collect_image_samples(
        data_root=TEST_ROOT,
    )

    # Train에는 랜덤 데이터 증강이 포함된 Transform을 연결한다.
    train_dataset = (
        CastingDefectDataset(
            samples=train_samples,
            transform=(
                create_train_transform()
            ),
        )
    )

    # Validation에는 랜덤 증강이 없는 평가 Transform을 연결한다.
    validation_dataset = (
        CastingDefectDataset(
            samples=validation_samples,
            transform=(
                create_validation_transform()
            ),
        )
    )

    # 기존 Test에도 랜덤 증강이 없는 최종 평가 Transform을 연결한다.
    test_dataset = (
        CastingDefectDataset(
            samples=test_samples,
            transform=(
                create_test_transform()
            ),
        )
    )

    # Train은 Epoch마다 Sample 순서를 섞는다.
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(
            persistent_workers
        ),
        random_seed=random_seed,
    )

    # Validation은 평가 순서와 이미지 Path 연결을
    # 일정하게 유지하기 위해 섞지 않는다.
    validation_loader = (
        create_data_loader(
            dataset=validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(
                persistent_workers
            ),
            random_seed=random_seed,
        )
    )

    # Test도 최종 Prediction·오분류 이미지 Path 연결을 위해
    # 고정 순서를 사용한다.
    test_loader = create_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(
            persistent_workers
        ),
        random_seed=random_seed,
    )

    return VisionDataLoaders(
        train_dataset=train_dataset,
        validation_dataset=(
            validation_dataset
        ),
        test_dataset=test_dataset,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        test_loader=test_loader,
    )


def validate_image_label_batch(
    images: Tensor,
    labels: Tensor,
) -> None:
    """
    DataLoader가 생성한 Image·Label Batch 구조를 검증한다.

    Parameters
    ----------
    images:
        DataLoader가 반환한 Image Tensor Batch다.

        기대 Shape:

            [Batch, 3, 224, 224]

    labels:
        DataLoader가 반환한 Label Tensor Batch다.

        기대 Shape:

            [Batch]

    Raises
    ------
    TypeError
        images 또는 labels가 Tensor가 아닐 때 발생한다.

    ValueError
        Batch 차원·Channel·공간 크기·dtype·Sample 수가
        기대 조건과 다를 때 발생한다.

    필요한 이유
    -----------
    Dataset 단계에서는 이미지 한 장을 검증했다.

        [3, 224, 224]

    DataLoader 단계에서는 Batch 차원이 추가된 구조를 검증한다.

        [Batch, 3, 224, 224]
    """

    if not isinstance(
        images,
        Tensor,
    ):
        raise TypeError(
            "images는 torch.Tensor여야 합니다: "
            f"{type(images).__name__}"
        )

    if not isinstance(
        labels,
        Tensor,
    ):
        raise TypeError(
            "labels는 torch.Tensor여야 합니다: "
            f"{type(labels).__name__}"
        )

    # Image Batch:
    #
    #     [Batch, Channel, Height, Width]
    #
    # 총 4차원이어야 한다.
    if images.ndim != 4:
        raise ValueError(
            "Image Batch는 4차원 "
            "[B, C, H, W]여야 합니다: "
            f"shape={tuple(images.shape)}"
        )

    if images.shape[0] <= 0:
        raise ValueError(
            "Image Batch Size는 1 이상이어야 합니다."
        )

    if images.shape[1:] != (
        3,
        224,
        224,
    ):
        raise ValueError(
            "Image Batch의 개별 이미지 Shape는 "
            "[3, 224, 224]여야 합니다: "
            f"shape={tuple(images.shape)}"
        )

    if not torch.is_floating_point(
        images
    ):
        raise ValueError(
            "Image Batch는 Floating Point "
            "Tensor여야 합니다: "
            f"dtype={images.dtype}"
        )

    if not torch.isfinite(
        images
    ).all().item():
        raise ValueError(
            "Image Batch에 NaN 또는 inf가 있습니다."
        )

    # Label Batch:
    #
    #     [Batch]
    #
    # 1차원이어야 한다.
    if labels.ndim != 1:
        raise ValueError(
            "Label Batch는 1차원 "
            "[B]여야 합니다: "
            f"shape={tuple(labels.shape)}"
        )

    # Dataset의 int Label을 DataLoader 기본 Collate가 묶으면
    # 일반적으로 torch.int64 Tensor가 된다.
    if labels.dtype != torch.int64:
        raise ValueError(
            "Label Batch dtype은 "
            "torch.int64여야 합니다: "
            f"dtype={labels.dtype}"
        )

    # 이미지 수와 Label 수가 일치해야 한다.
    if (
        images.shape[0]
        != labels.shape[0]
    ):
        raise ValueError(
            "Image Batch와 Label Batch의 "
            "Sample 수가 다릅니다. "
            f"image_count={images.shape[0]}, "
            f"label_count={labels.shape[0]}"
        )

    # 현재 이진 분류 Label은 0 또는 1만 허용한다.
    valid_label_mask = (
        (labels == 0)
        | (labels == 1)
    )

    if not valid_label_mask.all().item():
        raise ValueError(
            "Label Batch에는 0 또는 1만 "
            "포함되어야 합니다: "
            f"labels={labels.tolist()}"
        )


def print_data_loader_summary(
    data_loaders: VisionDataLoaders,
) -> None:
    """
    Dataset 수·Batch 수·DataLoader 설정을 출력한다.

    Parameters
    ----------
    data_loaders:
        Train·Validation·Test Dataset과 DataLoader 객체다.
    """

    print(
        "=" * 80
    )

    print(
        "DAY 2 - PYTORCH DATASET / "
        "DATALOADER SUMMARY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "[CONFIGURATION]"
    )

    print(
        f"batch size        : {BATCH_SIZE}"
    )

    print(
        f"num workers       : {NUM_WORKERS}"
    )

    print(
        f"pin memory        : {PIN_MEMORY}"
    )

    print(
        f"drop last         : {DROP_LAST}"
    )

    print(
        "persistent workers: "
        f"{PERSISTENT_WORKERS}"
    )

    print(
        f"random seed       : {RANDOM_SEED}"
    )

    print()

    print(
        "[DATASET COUNTS]"
    )

    print(
        "train            : "
        f"{len(data_loaders.train_dataset)}"
    )

    print(
        "validation       : "
        f"{len(data_loaders.validation_dataset)}"
    )

    print(
        "test             : "
        f"{len(data_loaders.test_dataset)}"
    )

    print()

    print(
        "[BATCH COUNTS]"
    )

    print(
        "train            : "
        f"{len(data_loaders.train_loader)}"
    )

    print(
        "validation       : "
        f"{len(data_loaders.validation_loader)}"
    )

    print(
        "test             : "
        f"{len(data_loaders.test_loader)}"
    )

    print()

    print(
        "[SAMPLER]"
    )

    print(
        "train            : "
        f"{type(data_loaders.train_loader.sampler).__name__}"
    )

    print(
        "validation       : "
        f"{type(data_loaders.validation_loader.sampler).__name__}"
    )

    print(
        "test             : "
        f"{type(data_loaders.test_loader.sampler).__name__}"
    )

    print()


def print_batch_summary(
    split_name: str,
    data_loader: DataLoader,
) -> None:
    """
    DataLoader의 첫 Batch Tensor 정보를 출력한다.

    Parameters
    ----------
    split_name:
        출력할 Split 이름이다.

        예:

            TRAIN

            VALIDATION

            TEST

    data_loader:
        첫 Batch를 확인할 DataLoader다.
    """

    images, labels = next(
        iter(data_loader)
    )

    validate_image_label_batch(
        images=images,
        labels=labels,
    )

    print(
        f"[{split_name} FIRST BATCH]"
    )

    print(
        "image shape : "
        f"{tuple(images.shape)}"
    )

    print(
        "label shape : "
        f"{tuple(labels.shape)}"
    )

    print(
        "image dtype : "
        f"{images.dtype}"
    )

    print(
        "label dtype : "
        f"{labels.dtype}"
    )

    print(
        "image min   : "
        f"{images.min().item():.6f}"
    )

    print(
        "image max   : "
        f"{images.max().item():.6f}"
    )

    print(
        "image finite: "
        f"{torch.isfinite(images).all().item()}"
    )

    print(
        "labels      : "
        f"{labels.tolist()}"
    )

    print()


def validate_sampler_types(
    data_loaders: VisionDataLoaders,
) -> None:
    """
    Train·Validation·Test DataLoader의 Sampler 구성을 검증한다.

    Train
    -----
    shuffle=True이므로:

        RandomSampler

    Validation·Test
    ----------------
    shuffle=False이므로:

        SequentialSampler
    """

    if not isinstance(
        data_loaders.train_loader.sampler,
        RandomSampler,
    ):
        raise ValueError(
            "Train DataLoader는 "
            "RandomSampler를 사용해야 합니다."
        )

    if not isinstance(
        data_loaders.validation_loader.sampler,
        SequentialSampler,
    ):
        raise ValueError(
            "Validation DataLoader는 "
            "SequentialSampler를 사용해야 합니다."
        )

    if not isinstance(
        data_loaders.test_loader.sampler,
        SequentialSampler,
    ):
        raise ValueError(
            "Test DataLoader는 "
            "SequentialSampler를 사용해야 합니다."
        )


def main() -> None:
    """
    실제 Casting 데이터로 Dataset·DataLoader를 구성하고
    첫 Batch를 검증한다.

    실행 명령
    ---------
    프로젝트 Root에서:

        python -m src.data.data_loader

    실행 순서
    ---------
    Dataset·DataLoader 전체 생성

    → Sampler 검증

    → Dataset 수 출력

    → Batch 수 출력

    → Train 첫 Batch 검증

    → Validation 첫 Batch 검증

    → Test 첫 Batch 검증
    """

    data_loaders = (
        create_vision_data_loaders()
    )

    validate_sampler_types(
        data_loaders=data_loaders,
    )

    print_data_loader_summary(
        data_loaders=data_loaders,
    )

    print_batch_summary(
        split_name="TRAIN",
        data_loader=(
            data_loaders.train_loader
        ),
    )

    print_batch_summary(
        split_name="VALIDATION",
        data_loader=(
            data_loaders.validation_loader
        ),
    )

    print_batch_summary(
        split_name="TEST",
        data_loader=(
            data_loaders.test_loader
        ),
    )


if __name__ == "__main__":
    main()
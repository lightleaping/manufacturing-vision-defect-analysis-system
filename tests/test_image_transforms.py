"""
Train·Validation·Test 이미지 Transform 테스트.

이 테스트 파일의 책임
---------------------
1. 이미지 입력 크기 설정을 검증한다.
2. ImageNet 평균·표준편차 설정을 검증한다.
3. Train Transform의 처리 순서와 랜덤 증강 구성을 검증한다.
4. Validation·Test Transform에 랜덤 증강이 없는지 검증한다.
5. Transform 결과의 Shape·dtype·유한값을 검증한다.
6. 평가 Transform이 같은 입력에 항상 같은 Tensor를 반환하는지 검증한다.
7. Validation과 Test가 같은 전처리 결과를 생성하는지 검증한다.
8. ImageNet Normalize 계산 결과를 검증한다.
9. Random Seed를 다시 설정했을 때 Train Transform 결과를 재현할 수 있는지 검증한다.
10. Transform이 원본 Pillow 이미지를 변경하지 않는지 검증한다.
11. 실제 Casting 이미지가 평가 Transform을 정상 통과하는지 검증한다.

중요
----
Train Transform에는 랜덤 증강이 포함된다.

따라서 단순히 두 번 실행한 결과가 반드시 달라야 한다고
검증하지 않는다.

랜덤 결과는 우연히 같을 수도 있기 때문이다.

대신 다음을 확인한다.

    Random Transform이 Pipeline에 포함되어 있는가?

    같은 Random Seed를 다시 적용하면 같은 결과가 생성되는가?
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import (
    Image,
    ImageDraw,
)
from torch import Tensor
from torchvision import transforms

from src.data.dataset_config import (
    TRAIN_ROOT,
)
from src.data.dataset_split import (
    collect_image_samples,
)
from src.data.image_dataset import (
    CastingDefectDataset,
)
from src.data.image_transforms import (
    HORIZONTAL_FLIP_PROBABILITY,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROTATION_DEGREES,
    create_evaluation_transform,
    create_test_transform,
    create_train_transform,
    create_validation_transform,
)


def create_asymmetric_test_image(
    size: tuple[int, int] = (
        80,
        40,
    ),
) -> Image.Image:
    """
    랜덤 반전·회전 테스트에 사용할 비대칭 RGB 이미지를 생성한다.

    Parameters
    ----------
    size:
        Pillow 이미지 크기다.

        Pillow 순서:

            Width, Height

        기본값:

            Width  = 80

            Height = 40

    Returns
    -------
    PIL.Image.Image
        좌우 모양이 서로 다른 RGB 이미지다.

    비대칭 이미지를 사용하는 이유
    -----------------------------
    이미지 전체가 한 가지 색이거나 좌우 대칭이면
    RandomHorizontalFlip이 적용되어도 결과가 같아 보일 수 있다.

    따라서 왼쪽과 오른쪽에 서로 다른 색상 영역을 만들어
    공간 Transform이 실제 이미지 구조를 처리하도록 한다.
    """

    image = Image.new(
        mode="RGB",
        size=size,
        color=(
            20,
            20,
            20,
        ),
    )

    image_draw = ImageDraw.Draw(
        image
    )

    width, height = size

    # 이미지 왼쪽에는 빨간 사각형을 생성한다.
    image_draw.rectangle(
        (
            5,
            5,
            width // 3,
            height - 5,
        ),
        fill=(
            255,
            0,
            0,
        ),
    )

    # 이미지 오른쪽에는 파란 사각형을 생성한다.
    image_draw.rectangle(
        (
            width * 2 // 3,
            10,
            width - 5,
            height - 10,
        ),
        fill=(
            0,
            0,
            255,
        ),
    )

    return image


def test_transform_constants_are_expected() -> None:
    """
    모델 입력 크기와 Transform 설정값이 프로젝트 기준과 일치하는지 검증한다.
    """

    assert IMAGE_SIZE == (
        224,
        224,
    )

    assert IMAGENET_MEAN == pytest.approx(
        (
            0.485,
            0.456,
            0.406,
        )
    )

    assert IMAGENET_STD == pytest.approx(
        (
            0.229,
            0.224,
            0.225,
        )
    )

    assert (
        HORIZONTAL_FLIP_PROBABILITY
        == pytest.approx(0.50)
    )

    assert (
        ROTATION_DEGREES
        == pytest.approx(5.0)
    )


def test_train_transform_contains_expected_steps_and_parameters() -> None:
    """
    Train Transform의 순서와 주요 설정을 검증한다.

    기대 순서
    ---------
    Resize

    → RandomHorizontalFlip

    → RandomRotation

    → ToTensor

    → Normalize
    """

    train_transform = (
        create_train_transform()
    )

    assert isinstance(
        train_transform,
        transforms.Compose,
    )

    transform_steps = (
        train_transform.transforms
    )

    assert len(
        transform_steps
    ) == 5

    assert isinstance(
        transform_steps[0],
        transforms.Resize,
    )

    assert isinstance(
        transform_steps[1],
        transforms.RandomHorizontalFlip,
    )

    assert isinstance(
        transform_steps[2],
        transforms.RandomRotation,
    )

    assert isinstance(
        transform_steps[3],
        transforms.ToTensor,
    )

    assert isinstance(
        transform_steps[4],
        transforms.Normalize,
    )

    resize_transform = (
        transform_steps[0]
    )

    horizontal_flip_transform = (
        transform_steps[1]
    )

    rotation_transform = (
        transform_steps[2]
    )

    normalize_transform = (
        transform_steps[4]
    )

    # torchvision 내부에서는 Resize Size가 list로 저장될 수도 있으므로
    # Tuple로 변환한 뒤 프로젝트 설정과 비교한다.
    assert tuple(
        resize_transform.size
    ) == IMAGE_SIZE

    assert (
        horizontal_flip_transform.p
        == pytest.approx(
            HORIZONTAL_FLIP_PROBABILITY
        )
    )

    # degrees=5.0을 전달하면 torchvision은
    # 내부적으로 [-5.0, +5.0] 범위로 저장한다.
    assert tuple(
        rotation_transform.degrees
    ) == pytest.approx(
        (
            -ROTATION_DEGREES,
            ROTATION_DEGREES,
        )
    )

    assert tuple(
        normalize_transform.mean
    ) == pytest.approx(
        IMAGENET_MEAN
    )

    assert tuple(
        normalize_transform.std
    ) == pytest.approx(
        IMAGENET_STD
    )


def test_evaluation_transform_contains_only_deterministic_steps() -> None:
    """
    평가 Transform이 랜덤 증강 없이 고정 전처리만 사용하는지 검증한다.

    기대 순서
    ---------
    Resize

    → ToTensor

    → Normalize
    """

    evaluation_transform = (
        create_evaluation_transform()
    )

    assert isinstance(
        evaluation_transform,
        transforms.Compose,
    )

    transform_steps = (
        evaluation_transform.transforms
    )

    assert len(
        transform_steps
    ) == 3

    assert isinstance(
        transform_steps[0],
        transforms.Resize,
    )

    assert isinstance(
        transform_steps[1],
        transforms.ToTensor,
    )

    assert isinstance(
        transform_steps[2],
        transforms.Normalize,
    )

    # Validation·Test 평가 입력은 실행할 때마다 같아야 하므로
    # 랜덤 좌우 반전이 없어야 한다.
    assert not any(
        isinstance(
            transform_step,
            transforms.RandomHorizontalFlip,
        )
        for transform_step in transform_steps
    )

    # 랜덤 회전도 없어야 한다.
    assert not any(
        isinstance(
            transform_step,
            transforms.RandomRotation,
        )
        for transform_step in transform_steps
    )


def test_transform_factories_return_independent_instances() -> None:
    """
    Transform 생성 함수를 호출할 때마다 새로운 객체가 생성되는지 검증한다.

    필요한 이유
    -----------
    Train·Validation·Test Dataset이 같은 Transform 객체를
    의도하지 않게 공유하지 않도록 각 호출에서 독립 객체를 생성한다.
    """

    first_train_transform = (
        create_train_transform()
    )

    second_train_transform = (
        create_train_transform()
    )

    validation_transform = (
        create_validation_transform()
    )

    test_transform = (
        create_test_transform()
    )

    assert (
        first_train_transform
        is not second_train_transform
    )

    assert (
        validation_transform
        is not test_transform
    )


def test_evaluation_transform_returns_expected_shape_dtype_and_finite_values() -> None:
    """
    평가 Transform 결과가 모델 입력 조건을 만족하는지 검증한다.

    기대 결과
    ---------
    Shape:

        [3, 224, 224]

    dtype:

        torch.float32

    값:

        NaN·inf 없음
    """

    image = create_asymmetric_test_image()

    evaluation_transform = (
        create_evaluation_transform()
    )

    image_tensor = (
        evaluation_transform(
            image
        )
    )

    assert isinstance(
        image_tensor,
        Tensor,
    )

    assert image_tensor.shape == (
        3,
        224,
        224,
    )

    assert image_tensor.dtype == (
        torch.float32
    )

    assert torch.isfinite(
        image_tensor
    ).all().item()


def test_evaluation_transform_is_deterministic() -> None:
    """
    같은 이미지에 평가 Transform을 반복 적용하면 같은 Tensor가 생성되는지 검증한다.
    """

    image = create_asymmetric_test_image()

    evaluation_transform = (
        create_evaluation_transform()
    )

    first_tensor = (
        evaluation_transform(
            image
        )
    )

    second_tensor = (
        evaluation_transform(
            image
        )
    )

    assert torch.equal(
        first_tensor,
        second_tensor,
    )


def test_validation_and_test_transforms_return_equal_tensors() -> None:
    """
    현재 Validation과 Test가 동일한 평가 전처리 결과를 만드는지 검증한다.
    """

    image = create_asymmetric_test_image()

    validation_transform = (
        create_validation_transform()
    )

    test_transform = (
        create_test_transform()
    )

    validation_tensor = (
        validation_transform(
            image
        )
    )

    test_tensor = (
        test_transform(
            image
        )
    )

    assert torch.equal(
        validation_tensor,
        test_tensor,
    )


def test_evaluation_normalization_uses_imagenet_statistics() -> None:
    """
    ImageNet 평균·표준편차가 실제 Normalize 계산에 적용되는지 검증한다.

    테스트 RGB 값
    ---------------
    Red:

        255

        → ToTensor 후 1.0

    Green:

        0

        → ToTensor 후 0.0

    Blue:

        128

        → ToTensor 후 128 / 255

    기대 계산
    ---------
    normalized

    =

    (pixel - mean)

    / std
    """

    image = Image.new(
        mode="RGB",
        size=(
            16,
            16,
        ),
        color=(
            255,
            0,
            128,
        ),
    )

    evaluation_transform = (
        create_evaluation_transform()
    )

    image_tensor = (
        evaluation_transform(
            image
        )
    )

    red_value = 1.0

    green_value = 0.0

    blue_value = (
        128.0
        / 255.0
    )

    expected_channel_values = (
        torch.tensor(
            [
                (
                    red_value
                    - IMAGENET_MEAN[0]
                )
                / IMAGENET_STD[0],
                (
                    green_value
                    - IMAGENET_MEAN[1]
                )
                / IMAGENET_STD[1],
                (
                    blue_value
                    - IMAGENET_MEAN[2]
                )
                / IMAGENET_STD[2],
            ],
            dtype=torch.float32,
        )
    )

    actual_channel_values = (
        image_tensor[
            :,
            0,
            0,
        ]
    )

    assert torch.allclose(
        actual_channel_values,
        expected_channel_values,
        atol=1e-5,
        rtol=1e-5,
    )


def test_train_transform_returns_expected_shape_dtype_and_finite_values() -> None:
    """
    랜덤 증강이 포함된 Train Transform도 모델 입력 조건을 유지하는지 검증한다.
    """

    image = create_asymmetric_test_image()

    train_transform = (
        create_train_transform()
    )

    image_tensor = (
        train_transform(
            image
        )
    )

    assert isinstance(
        image_tensor,
        Tensor,
    )

    assert image_tensor.shape == (
        3,
        224,
        224,
    )

    assert image_tensor.dtype == (
        torch.float32
    )

    assert torch.isfinite(
        image_tensor
    ).all().item()


def test_train_transform_is_reproducible_when_seed_is_reset() -> None:
    """
    같은 Random Seed를 다시 적용하면 Train Transform 결과가 재현되는지 검증한다.

    중요
    ----
    Train Transform은 원래 랜덤 증강을 사용한다.

    항상 같은 Seed를 사용하는 것이 학습 목표는 아니다.

    이 테스트는 Random Seed를 명시적으로 다시 설정했을 때
    같은 랜덤 연산을 재현할 수 있는지만 확인한다.
    """

    image = create_asymmetric_test_image()

    train_transform = (
        create_train_transform()
    )

    random_seed = 42

    torch.manual_seed(
        random_seed
    )

    first_tensor = (
        train_transform(
            image.copy()
        )
    )

    torch.manual_seed(
        random_seed
    )

    second_tensor = (
        train_transform(
            image.copy()
        )
    )

    assert torch.equal(
        first_tensor,
        second_tensor,
    )


def test_transforms_do_not_modify_original_pillow_image() -> None:
    """
    Transform 실행 후에도 원본 Pillow 이미지의 Mode와 Size가 유지되는지 검증한다.

    필요한 이유
    -----------
    Transform은 모델 입력 Tensor를 생성해야 하며,
    원본 이미지 파일이나 원본 Pillow 객체의 기본 정보를
    의도하지 않게 변경하면 안 된다.
    """

    image = create_asymmetric_test_image(
        size=(
            80,
            40,
        )
    )

    original_mode = (
        image.mode
    )

    original_size = (
        image.size
    )

    train_transform = (
        create_train_transform()
    )

    evaluation_transform = (
        create_evaluation_transform()
    )

    _ = train_transform(
        image
    )

    _ = evaluation_transform(
        image
    )

    assert image.mode == (
        original_mode
    )

    assert image.size == (
        original_size
    )


def test_real_casting_image_passes_evaluation_transform() -> None:
    """
    실제 Casting 이미지가 Dataset과 평가 Transform을 정상 통과하는지 검증한다.

    데이터 흐름
    -----------
    실제 이미지 Path

    → ImageSample

    → CastingDefectDataset

    → Image.open()

    → RGB

    → Validation Transform

    → Tensor [3, 224, 224]

    → Label
    """

    all_samples = (
        collect_image_samples(
            data_root=TRAIN_ROOT,
        )
    )

    first_sample = (
        all_samples[0]
    )

    dataset = (
        CastingDefectDataset(
            samples=[
                first_sample,
            ],
            transform=(
                create_validation_transform()
            ),
        )
    )

    image_tensor, label = (
        dataset[0]
    )

    assert image_tensor.shape == (
        3,
        224,
        224,
    )

    assert image_tensor.dtype == (
        torch.float32
    )

    assert torch.isfinite(
        image_tensor
    ).all().item()

    assert label == 0

    assert (
        dataset.samples[0].image_path
        == first_sample.image_path
    )
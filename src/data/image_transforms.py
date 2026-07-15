"""
Train·Validation·Test 이미지 Transform 모듈.

이 모듈의 책임
----------------
1. 모델 입력 이미지 크기를 224 × 224로 통일한다.
2. Train 데이터에만 보수적인 랜덤 데이터 증강을 적용한다.
3. Pillow 이미지를 PyTorch float32 Tensor로 변환한다.
4. ImageNet 평균·표준편차를 사용하여 RGB Channel을 정규화한다.
5. Validation과 Test에는 랜덤 증강이 없는 동일한 평가 Transform을 제공한다.

데이터 흐름
----------
Train:

    PIL RGB Image

    → Resize

    → RandomHorizontalFlip

    → RandomRotation

    → ToTensor

    → Normalize

    → Tensor [3, 224, 224]

Validation·Test:

    PIL RGB Image

    → Resize

    → ToTensor

    → Normalize

    → Tensor [3, 224, 224]

중요
----
Train·Validation 분리는 이 모듈이 담당하지 않는다.

Train·Validation 분리:

    src.data.dataset_split

이미지 파일 로드와 RGB 변환:

    src.data.image_dataset

DataLoader 생성:

    이후 별도 모듈에서 구현
"""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import (
    InterpolationMode,
)


# CNN Baseline과 ResNet18에 전달할 이미지 크기다.
#
# torchvision Resize의 Tuple 순서:
#
#     Height, Width
#
# 현재는 정사각형 입력을 사용한다.
IMAGE_SIZE: tuple[int, int] = (
    224,
    224,
)


# ImageNet 사전학습 모델에서 사용하는 RGB Channel 평균이다.
#
# 순서:
#
#     Red
#     Green
#     Blue
#
# 이후 ResNet18 전이학습과 호환하기 위해 사용한다.
IMAGENET_MEAN: tuple[
    float,
    float,
    float,
] = (
    0.485,
    0.456,
    0.406,
)


# ImageNet 사전학습 모델에서 사용하는 RGB Channel 표준편차다.
#
# 순서:
#
#     Red
#     Green
#     Blue
IMAGENET_STD: tuple[
    float,
    float,
    float,
] = (
    0.229,
    0.224,
    0.225,
)


# Train 이미지에 좌우 반전을 적용할 확률이다.
#
# 0.50:
#
#     약 50% 확률로 좌우 반전
#
# Validation과 Test에는 사용하지 않는다.
HORIZONTAL_FLIP_PROBABILITY: float = 0.50


# Train 이미지에 적용할 최대 회전 각도다.
#
# degrees=5.0:
#
#     약 -5도 ~ +5도 범위에서 랜덤 회전
#
# 큰 회전은 실제 제조 이미지 분포를 지나치게 변경하거나
# 불필요한 경계 Artifact를 만들 수 있으므로 작은 값만 사용한다.
ROTATION_DEGREES: float = 5.0


def create_train_transform() -> transforms.Compose:
    """
    Train Dataset에 사용할 이미지 Transform을 생성한다.

    Returns
    -------
    torchvision.transforms.Compose
        다음 순서의 Transform Pipeline이다.

        1. Resize

            원본 300 × 300

            → 224 × 224

        2. RandomHorizontalFlip

            약 50% 확률로 좌우 반전

        3. RandomRotation

            약 -5도 ~ +5도 범위의 작은 회전

        4. ToTensor

            H × W × C

            → C × H × W

            0 ~ 255

            → 0.0 ~ 1.0

        5. Normalize

            ImageNet RGB 평균·표준편차 사용

    Train에만 랜덤 증강을 적용하는 이유
    ----------------------------------
    모델이 학습 이미지의 정확한 위치나 방향을 단순 암기하지 않고,
    작은 위치·방향 변화에도 정상·불량 특징을 인식하도록 돕기 위해서다.

    중요
    ----
    이 함수는 호출할 때마다 새로운 Compose 객체를 생성한다.

    Dataset 객체 간에 Transform Instance를 불필요하게 공유하지 않고
    각 Dataset 구성을 독립적으로 확인할 수 있다.
    """

    return transforms.Compose(
        [
            # 모든 모델 입력 이미지의 크기를 동일하게 만든다.
            #
            # antialias=True:
            #
            # 이미지 축소 과정에서 계단 현상과 고주파 Artifact를
            # 줄이기 위한 Antialias 처리를 사용한다.
            transforms.Resize(
                size=IMAGE_SIZE,
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            ),

            # 약 50% 확률로 이미지를 좌우 반전한다.
            #
            # 상하 반전은 현재 적용하지 않는다.
            #
            # 제조 이미지의 실제 촬영 방향과 지나치게 다른 입력을
            # 만들 가능성을 줄이기 위해 보수적으로 구성한다.
            transforms.RandomHorizontalFlip(
                p=(
                    HORIZONTAL_FLIP_PROBABILITY
                ),
            ),

            # 작은 촬영 각도 차이에 대응하기 위한 회전 증강이다.
            #
            # degrees=5.0:
            #
            #     -5도 ~ +5도
            #
            # BILINEAR 보간을 사용하여 회전 후 픽셀값을 계산한다.
            #
            # 이미지 밖 영역은 0으로 채운다.
            transforms.RandomRotation(
                degrees=ROTATION_DEGREES,
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                fill=0,
            ),

            # Pillow 이미지를 PyTorch Tensor로 변환한다.
            #
            # 변환 전:
            #
            #     [Height, Width, Channel]
            #
            # 변환 후:
            #
            #     [Channel, Height, Width]
            #
            # 일반적인 8-bit 이미지:
            #
            #     0 ~ 255
            #
            # float32 Tensor:
            #
            #     0.0 ~ 1.0
            transforms.ToTensor(),

            # RGB Channel별 ImageNet 평균·표준편차를 사용한다.
            #
            # 계산:
            #
            #     normalized
            #
            #     =
            #
            #     (pixel - mean)
            #
            #     / std
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def create_evaluation_transform() -> (
    transforms.Compose
):
    """
    Validation·Test 공통 평가 Transform을 생성한다.

    Returns
    -------
    torchvision.transforms.Compose
        다음 순서의 Transform Pipeline이다.

        1. Resize

        2. ToTensor

        3. Normalize

    랜덤 증강을 제외하는 이유
    -----------------------
    Validation과 Test는 같은 이미지에 대해 항상 같은 입력 Tensor를
    생성해야 한다.

    모델이 변경되지 않았는데 평가 시점마다 이미지 회전·반전이 달라지면
    성능 비교의 일관성이 낮아질 수 있다.

    따라서 평가 Transform에는 다음을 포함하지 않는다.

        RandomHorizontalFlip

        RandomRotation
    """

    return transforms.Compose(
        [
            transforms.Resize(
                size=IMAGE_SIZE,
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def create_validation_transform() -> (
    transforms.Compose
):
    """
    Validation Dataset용 Transform을 생성한다.

    Returns
    -------
    torchvision.transforms.Compose
        랜덤 증강이 없는 평가용 Transform이다.

    별도 함수로 제공하는 이유
    -----------------------
    현재 Validation과 Test의 전처리는 동일하다.

    하지만 역할은 서로 다르다.

    Validation:

        학습 중 모델 상태 확인

        모델 선택

        과적합 확인

    Test:

        최종 모델 성능 평가

    호출 이름을 분리하면 Dataset 구성 코드에서 각 데이터의 역할을
    더 명확하게 표현할 수 있다.
    """

    return create_evaluation_transform()


def create_test_transform() -> transforms.Compose:
    """
    Test Dataset용 Transform을 생성한다.

    Returns
    -------
    torchvision.transforms.Compose
        랜덤 증강이 없는 최종 평가용 Transform이다.

    중요
    ----
    Test 데이터는 최종 모델 평가 전용이다.

    Test Transform 결과를 반복 확인하면서 모델 구조나 학습 설정을
    조정하지 않는다.
    """

    return create_evaluation_transform()
"""
PyTorch 이미지 Dataset 모듈.

이 모듈의 책임
----------------
1. Train·Validation·Test에 사용할 ImageSample 목록을 저장한다.
2. 요청된 인덱스의 이미지 파일을 Pillow로 읽는다.
3. 모든 이미지를 RGB 3채널로 변환한다.
4. 외부에서 전달받은 Transform을 이미지에 적용한다.
5. 모델 입력에 사용할 Image Tensor와 정수 Label을 반환한다.
6. 잘못된 Path, Label, Transform 결과를 명확한 예외로 처리한다.

데이터 흐름
----------
ImageSample

    image_path
    label

→ CastingDefectDataset

→ __getitem__(index)

→ Pillow Image.open()

→ RGB 변환

→ Transform

→ Image Tensor

→ (Image Tensor, Label) 반환

중요
----
이 모듈은 Train·Validation 분리를 수행하지 않는다.

Train·Validation 분리는 다음 모듈이 담당한다.

    src.data.dataset_split

이 모듈은 DataLoader도 생성하지 않는다.

DataLoader 구성은 이후 별도 모듈에서 구현한다.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Collection,
    Sequence,
)
from pathlib import Path

import torch
from PIL import (
    Image,
    UnidentifiedImageError,
)
from torch import Tensor
from torch.utils.data import Dataset

from src.data.dataset_config import (
    CLASS_TO_INDEX,
)
from src.data.dataset_split import (
    ImageSample,
)


# Transform은 Pillow Image를 입력받아 PyTorch Tensor를 반환한다.
#
# 이후 구현할 Train·Validation·Test Transform도 이 호출 구조를 따른다.
#
# 예:
#
#     image_tensor = transform(image)
#
ImageTransform = Callable[
    [Image.Image],
    Tensor,
]


# 현재 프로젝트에서 허용하는 Label 집합이다.
#
# CLASS_TO_INDEX:
#
#     {
#         "ok_front": 0,
#         "def_front": 1,
#     }
#
# 따라서:
#
#     ALLOWED_LABELS
#
#     → {0, 1}
#
# frozenset은 생성 후 값을 수정할 수 없는 불변 집합이다.
ALLOWED_LABELS: frozenset[int] = frozenset(
    CLASS_TO_INDEX.values()
)


class CastingDefectDataset(
    Dataset[tuple[Tensor, int]]
):
    """
    Casting 정상·불량 이미지용 PyTorch Dataset.

    Parameters
    ----------
    samples:
        이미지 Path와 Label을 저장한 ImageSample 목록이다.

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

    transform:
        Pillow Image를 PyTorch Tensor로 변환하는 함수 또는 Transform 객체다.

        이후 다음 Transform을 연결할 예정이다.

        Train:

            Resize
            Random Augmentation
            ToTensor
            Normalize

        Validation·Test:

            Resize
            ToTensor
            Normalize

    allowed_labels:
        현재 Dataset에서 허용할 정수 Label 집합이다.

        기본값:

            {0, 1}

    Dataset 반환
    -------------
    __getitem__()은 다음 Tuple을 반환한다.

        (
            image_tensor,
            label,
        )

    예:

        image_tensor.shape

            torch.Size(
                [3, 224, 224]
            )

        label

            0

        또는:

            1
    """

    def __init__(
        self,
        samples: Sequence[ImageSample],
        transform: ImageTransform,
        allowed_labels: Collection[int] = (
            ALLOWED_LABELS
        ),
    ) -> None:
        """
        Dataset을 초기화하고 입력 Sample을 검증한다.

        호출 시점
        ---------
        Dataset 객체를 생성할 때 한 번 실행된다.

        예:

            train_dataset = (
                CastingDefectDataset(
                    samples=train_samples,
                    transform=train_transform,
                )
            )

        처리
        ----
        1. Sample이 비어 있지 않은지 확인한다.
        2. Transform이 호출 가능한 객체인지 확인한다.
        3. 허용 Label 집합이 비어 있지 않은지 확인한다.
        4. Sample 목록을 Tuple로 저장한다.
        5. 이미지 Path와 Label을 검증한다.

        중요
        ----
        __init__()에서는 이미지 내용을 메모리에 모두 읽지 않는다.

        이미지 Path와 Label만 저장한다.

        실제 이미지는 __getitem__()이 호출될 때 한 장씩 읽는다.
        """

        # Dataset에 Sample이 하나도 없으면 모델 학습이나 평가를
        # 수행할 수 없으므로 즉시 오류를 발생시킨다.
        if not samples:
            raise ValueError(
                "Dataset Sample이 비어 있습니다."
            )

        # Dataset은 이미지에 Transform을 적용하여 Tensor를 반환한다.
        #
        # 함수, torchvision.transforms.Compose 등 호출 가능한 객체만
        # Transform으로 사용할 수 있다.
        if not callable(transform):
            raise TypeError(
                "transform은 호출 가능한 객체여야 합니다."
            )

        # 허용 Label을 수정할 수 없는 frozenset으로 변환한다.
        normalized_allowed_labels = frozenset(
            allowed_labels
        )

        if not normalized_allowed_labels:
            raise ValueError(
                "allowed_labels가 비어 있습니다."
            )

        # 외부에서 전달받은 list가 이후 수정되어 Dataset 구성이
        # 의도치 않게 변경되지 않도록 Tuple로 복사해 저장한다.
        self._samples: tuple[
            ImageSample,
            ...
        ] = tuple(
            samples
        )

        self._transform = transform

        self._allowed_labels = (
            normalized_allowed_labels
        )

        # Dataset 생성 시 Path·Label 구조를 먼저 검증한다.
        #
        # 실제 이미지 Binary의 정상 여부는 Day 1 분석과
        # __getitem__()의 Image.open() 단계에서 확인한다.
        self._validate_samples()

    def __len__(self) -> int:
        """
        Dataset의 전체 Sample 수를 반환한다.

        Returns
        -------
        int
            Dataset에 포함된 이미지 Sample 수다.

        예
        --
        현재 Train Dataset:

            len(train_dataset)

            → 5,306

        현재 Validation Dataset:

            len(validation_dataset)

            → 1,327

        필요한 이유
        -----------
        DataLoader는 Dataset 길이를 이용해 전체 Batch 수를 계산한다.

        Batch Size가 32이고 Dataset이 5,306장이라면,
        drop_last=False 기준 마지막 작은 Batch까지 포함한다.
        """

        return len(
            self._samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, int]:
        """
        특정 인덱스의 이미지 Tensor와 Label을 반환한다.

        Parameters
        ----------
        index:
            가져올 Sample의 위치다.

            예:

                dataset[0]

                → 첫 번째 이미지

                dataset[10]

                → 열한 번째 이미지

        Returns
        -------
        tuple[Tensor, int]
            첫 번째 값:

                Transform이 적용된 이미지 Tensor

            두 번째 값:

                정수 Label

                0 → NORMAL

                1 → DEFECT

        처리 순서
        ---------
        index

        → ImageSample 조회

        → Image.open()

        → RGB 변환

        → Transform

        → Tensor 구조 검증

        → image_tensor, label 반환

        Raises
        ------
        TypeError
            index가 정수가 아닐 때 발생한다.

            Transform 결과가 Tensor가 아닐 때 발생한다.

        IndexError
            index가 Dataset 범위를 벗어날 때 발생한다.

        RuntimeError
            이미지 파일을 열거나 읽을 수 없을 때 발생한다.

        ValueError
            Transform 결과 Tensor의 차원·채널·dtype이
            모델 입력 조건과 맞지 않을 때 발생한다.
        """

        # PyTorch DataLoader는 일반적으로 정수 Index를 전달한다.
        #
        # 잘못된 타입이 들어오면 list 내부 오류 대신 더 명확한
        # Dataset 오류를 제공한다.
        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "Dataset index는 int여야 합니다: "
                f"{type(index).__name__}"
            )

        # 음수 Index를 허용하면 Python list처럼 뒤에서부터 접근할 수
        # 있지만, 현재 Dataset에서는 명확한 0 이상 Index만 허용한다.
        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(
                "Dataset index가 범위를 벗어났습니다. "
                f"index={index}, "
                f"dataset_size={len(self)}"
            )

        sample = self._samples[
            index
        ]

        # Pillow Image.open()은 파일을 즉시 완전히 읽지 않고
        # 필요할 때 내용을 읽는 지연 로딩 방식을 사용한다.
        #
        # with 문을 사용하면 파일 Resource가 작업 후 안전하게 닫힌다.
        try:
            with Image.open(
                sample.image_path
            ) as image:
                # 현재 Day 1 데이터는 모두 RGB로 확인되었다.
                #
                # 그래도 convert("RGB")를 명시하여 향후 Grayscale,
                # RGBA 이미지가 추가되어도 모델 입력 채널을 항상
                # 3채널로 유지한다.
                rgb_image = image.convert(
                    "RGB"
                )

        except (
            FileNotFoundError,
            PermissionError,
            UnidentifiedImageError,
            OSError,
        ) as error:
            # Path를 포함한 프로젝트 수준 오류로 변환한다.
            #
            # 학습 도중 어느 이미지에서 문제가 발생했는지 바로
            # 확인할 수 있도록 원본 Path를 메시지에 포함한다.
            raise RuntimeError(
                "이미지를 읽을 수 없습니다: "
                f"{sample.image_path}"
            ) from error

        # 실제 Resize, ToTensor, Normalize 등의 처리는 Dataset이
        # 직접 결정하지 않고 외부 Transform 객체에 위임한다.
        #
        # 이렇게 하면 같은 Dataset 구조에 Train·Validation·Test용
        # Transform을 각각 연결할 수 있다.
        image_tensor = self._transform(
            rgb_image
        )

        # 모델 입력은 PyTorch Tensor여야 한다.
        #
        # Transform이 실수로 Pillow Image, NumPy Array 등을 반환하면
        # 모델 실행 전에 Dataset 단계에서 오류를 발견한다.
        if not isinstance(
            image_tensor,
            Tensor,
        ):
            raise TypeError(
                "Transform 결과는 torch.Tensor여야 합니다: "
                f"{type(image_tensor).__name__}"
            )

        # 이미지 한 장은 다음 3차원 구조여야 한다.
        #
        #     [Channel, Height, Width]
        #
        # Batch 차원은 이후 DataLoader가 추가한다.
        if image_tensor.ndim != 3:
            raise ValueError(
                "이미지 Tensor는 3차원 "
                "[C, H, W]여야 합니다: "
                f"shape={tuple(image_tensor.shape)}"
            )

        # 현재 프로젝트는 RGB 3채널 이미지 분류다.
        if image_tensor.shape[0] != 3:
            raise ValueError(
                "이미지 Tensor의 Channel은 "
                "3이어야 합니다: "
                f"shape={tuple(image_tensor.shape)}"
            )

        # CNN과 ResNet18은 Floating Point Tensor를 입력으로 사용한다.
        #
        # ToTensor()는 일반적인 8-bit 이미지 픽셀을 float Tensor로
        # 변환하므로 이후 정상적으로 통과한다.
        if not torch.is_floating_point(
            image_tensor
        ):
            raise ValueError(
                "이미지 Tensor는 Floating Point "
                "dtype이어야 합니다: "
                f"dtype={image_tensor.dtype}"
            )

        return (
            image_tensor,
            sample.label,
        )

    @property
    def samples(
        self,
    ) -> tuple[ImageSample, ...]:
        """
        Dataset이 사용하는 ImageSample을 읽기 전용 Tuple로 반환한다.

        Returns
        -------
        tuple[ImageSample, ...]
            Dataset 생성 시 저장한 Sample 목록이다.

        필요한 이유
        -----------
        이후 다음 작업에서 원본 이미지 Path를 확인할 수 있다.

            오분류 이미지 분석

            Grad-CAM 결과 저장

            Dataset 분리 확인

        Tuple로 반환하므로 외부 코드가 append() 또는 remove()로
        Dataset 구성을 직접 변경할 수 없다.
        """

        return self._samples

    @property
    def allowed_labels(
        self,
    ) -> frozenset[int]:
        """
        Dataset에서 허용하는 Label 집합을 반환한다.

        Returns
        -------
        frozenset[int]
            현재 기본값:

                {
                    0,
                    1,
                }
        """

        return self._allowed_labels

    def _validate_samples(
        self,
    ) -> None:
        """
        Dataset 생성 시 모든 ImageSample의 Path와 Label을 검증한다.

        검증 항목
        ---------
        1. 각 항목이 ImageSample인지 확인한다.
        2. Label이 int인지 확인한다.
        3. Label이 허용 Label 집합에 포함되는지 확인한다.
        4. 이미지 Path가 존재하는지 확인한다.
        5. 이미지 Path가 파일인지 확인한다.
        6. 같은 이미지 Path가 중복되지 않았는지 확인한다.

        중요
        ----
        여기서는 Image.open()으로 이미지 Binary를 읽지 않는다.

        Day 1에서 전체 이미지 품질을 이미 검증했으며,
        실제 이미지 읽기는 __getitem__()에서 수행한다.
        """

        normalized_path_keys: set[
            str
        ] = set()

        for sample_index, sample in enumerate(
            self._samples
        ):
            if not isinstance(
                sample,
                ImageSample,
            ):
                raise TypeError(
                    "Dataset의 모든 항목은 "
                    "ImageSample이어야 합니다. "
                    f"index={sample_index}, "
                    f"type={type(sample).__name__}"
                )

            # bool은 Python에서 int의 하위 타입이므로
            # type(...) is int를 사용해 실제 정수 Label만 허용한다.
            if type(sample.label) is not int:
                raise TypeError(
                    "ImageSample label은 int여야 합니다. "
                    f"index={sample_index}, "
                    f"label={sample.label!r}"
                )

            if (
                sample.label
                not in self._allowed_labels
            ):
                raise ValueError(
                    "허용되지 않은 Label입니다. "
                    f"index={sample_index}, "
                    f"label={sample.label}, "
                    "allowed_labels="
                    f"{sorted(self._allowed_labels)}"
                )

            image_path = Path(
                sample.image_path
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    "이미지 파일을 찾을 수 없습니다. "
                    f"index={sample_index}, "
                    f"path={image_path}"
                )

            if not image_path.is_file():
                raise ValueError(
                    "이미지 Path는 파일이어야 합니다. "
                    f"index={sample_index}, "
                    f"path={image_path}"
                )

            # Windows에서는 일반적으로 Path 대소문자를 구분하지 않는다.
            #
            # resolve():
            #
            #     절대 Path로 변환
            #
            # casefold():
            #
            #     대소문자 차이를 제거
            normalized_path_key = str(
                image_path.resolve()
            ).casefold()

            if (
                normalized_path_key
                in normalized_path_keys
            ):
                raise ValueError(
                    "Dataset에 중복 이미지 Path가 있습니다. "
                    f"index={sample_index}, "
                    f"path={image_path}"
                )

            normalized_path_keys.add(
                normalized_path_key
            )